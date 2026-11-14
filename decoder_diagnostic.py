"""L: measure saved decoder weights and variation across fixed empirical latent codes.

No training fallback is provided. Missing saved weights are reported as unexecuted.
"""
import argparse
import csv
import json
import os

from filelock import FileLock
import numpy as np
import torch

from collapse_replay import CACHE, test_batches
from diagnose_steps import build_model
from evaluation import ImageMetrics
from run_sweep import ROOT, RUNS, digest, identity, read_results, runtime_info
from training import Config, _parameter_digest, resolve_device
from untrained_encoder import experiment_spec as initial_spec, PROTOCOL as INITIAL_PROTOCOL

PROTOCOL = "circle-ae-decoder-diagnostic-v1"
LEDGER = ROOT / "results/decoder_diagnostic.jsonl"
CSV = ROOT / "results/decoder_diagnostic.csv"
GPU_LOCK = ROOT / "checkpoints/final_jkl_gpu.lock"
NORM_DEFINITION = "sqrt(sum(flatten(weight)**2)) in float64; Frobenius for tensors, L2 for vectors"
VARIATION_DEFINITION = "mean over 2*64*64 pixels of population SD across 5000 decoded empirical z values; float64 streaming Welford, denominator 5000"


class PixelPopulationMoments:
    """Each spatial/channel coordinate is a separate variable; only samples vary."""
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = None

    def update(self, predictions):
        values = np.asarray(predictions, dtype=np.float64)
        if values.ndim != 4 or values.shape[1:] != (2, 64, 64) or not len(values):
            raise ValueError("Expected a nonempty (samples, 2, 64, 64) prediction batch")
        if not np.isfinite(values).all():
            raise ValueError("Predictions must be finite")
        count = len(values)
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_m2 = np.square(values - batch_mean).sum(axis=0, dtype=np.float64)
        if not self.n:
            self.n, self.mean, self.m2 = count, batch_mean, batch_m2
            return
        total = self.n + count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * (self.n * count / total)
        self.mean += delta * (count / total)
        self.n = total

    def result(self):
        if not self.n:
            raise ValueError("No empirical latent codes were decoded")
        return float(np.sqrt(self.m2 / self.n).mean(dtype=np.float64))


def parameter_norms(model):
    modules = [("from_latent", model.from_latent),
               *(("decoder." + name, layer) for name, layer in model.decoder.named_children())]
    result = {}
    for name, layer in modules:
        parameters = dict(layer.named_parameters(recurse=False))
        if not parameters:
            continue
        if "weight" not in parameters or set(parameters) - {"weight", "bias"}:
            raise ValueError(f"Unrecognized parameterized decoder layer: {name}")

        def norm(tensor):
            if tensor is None:
                return None
            value = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            return float(np.sqrt(np.square(value).sum(dtype=np.float64)))

        weight = parameters["weight"]
        result[name] = dict(layer=name, layer_type=type(layer).__name__, weight_shape=list(weight.shape),
                            weight_numel=weight.numel(), weight_frobenius=norm(weight),
                            bias_l2=norm(parameters.get("bias")),
                            weight_kind="affine scale vector" if isinstance(layer, torch.nn.GroupNorm) else "kernel or matrix")
    return result


def design(rows, runtime):
    plan = []
    for loss in ("l1", "l2"):
        for seed in range(5):
            cfg = Config(radius=5, loss=loss, seed=seed, steps=3000, frame_gap=4,
                         reject_reflections=True, lr=.003, latent_dim=16)
            key = digest(identity(cfg, "experiment", runtime))
            source = rows[key]
            ikey = digest(initial_spec(cfg, {**runtime, "protocol": INITIAL_PROTOCOL}))
            initial = rows[ikey]
            if source["probe_splits"] != [6, 7] or initial["probe_splits"] != [6, 7]:
                raise ValueError("Unexpected probe split in the archived reference")
            plan.append((cfg, source, initial))
    return plan


def spec_for(source, initial):
    return {"purpose": "decoder_diagnostic", "config": source["config"],
            "runtime": {**source["runtime"], "protocol": PROTOCOL,
                        "source_config_hash": source["config_hash"], "initial_config_hash": initial["config_hash"]}}


def measure(cfg, source, initial, device):
    checkpoint = CACHE / source["config_hash"] / "weights.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = build_model(cfg, device)
    if _parameter_digest(model) != initial["model_sha256"]:
        raise ArithmeticError("Current default initialization differs from the archived zero-step encoder")
    before = parameter_norms(model)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["source_config_hash"] != source["config_hash"]:
        raise ValueError("Saved weights name a different archived experiment")
    model.load_state_dict(saved["model"])
    if _parameter_digest(model) != source["model_sha256"]:
        raise ArithmeticError("Saved model SHA differs from the archived experiment")
    after = parameter_norms(model)
    if before.keys() != after.keys():
        raise ValueError("Initialization and trained decoder layers do not match")
    layers = []
    for name in before:
        a, b = before[name], after[name]
        if any(a[k] != b[k] for k in ("layer_type", "weight_shape", "weight_numel", "weight_kind")):
            raise ValueError("Decoder architecture changed")
        layers.append({k: a[k] for k in ("layer", "layer_type", "weight_shape", "weight_numel", "weight_kind")} |
                      dict(initial_weight_frobenius=a["weight_frobenius"], trained_weight_frobenius=b["weight_frobenius"],
                           weight_trained_over_initial=b["weight_frobenius"] / a["weight_frobenius"] if a["weight_frobenius"] else None,
                           initial_bias_l2=a["bias_l2"], trained_bias_l2=b["bias_l2"]))
    model.eval().requires_grad_(False)
    images = ImageMetrics(include_center_intensity=True)
    variation = PixelPopulationMoments()
    # model(x) is decode(encode(x)); each of these fixed input examples supplies one z.
    for batch, prediction in test_batches(model, cfg, device):
        images.update(prediction, batch, cfg.radius)
        variation.update(prediction)
    metrics = images.result()
    if any(source["metrics"][key] != value for key, value in metrics.items()):
        raise ArithmeticError("Saved model image metrics do not exactly reproduce the original split 7 evaluation")
    if variation.n != 5000 or _parameter_digest(model) != source["model_sha256"]:
        raise ArithmeticError("Wrong evaluation count or model changed during inference")
    spec = spec_for(source, initial)
    return {**spec, "config_hash": digest(spec), "source_config_hash": source["config_hash"],
            "initial_config_hash": initial["config_hash"], "model_sha256": source["model_sha256"],
            "initial_model_sha256": initial["model_sha256"], "initialization": "default",
            "weights_path": str(checkpoint.relative_to(ROOT)), "weights_only": True,
            "initial_model_exact": True, "original_image_metrics_exact": True,
            "evaluation_seed_sequence": [7, 0], "evaluation_pairs": variation.n,
            "latent_source": "frozen encoder applied to the fixed clean evaluation inputs",
            "norm_definition": NORM_DEFINITION, "variation_definition": VARIATION_DEFINITION,
            "mean_pixel_population_sd_across_samples": variation.result(),
            "image_metrics": metrics, "layers": layers}


def export(plan, measured, path=CSV):
    fields = ("status", "loss", "seed", "source_config_hash", "initial_config_hash", "model_sha256",
              "initial_model_sha256", "mean_pixel_population_sd_across_samples", "layer", "layer_type",
              "weight_kind", "weight_shape", "weight_numel", "initial_weight_frobenius",
              "trained_weight_frobenius", "weight_trained_over_initial", "initial_bias_l2", "trained_bias_l2",
              "config_hash")
    records = []
    for cfg, source, initial in plan:
        key = digest(spec_for(source, initial))
        common = dict(loss=cfg.loss, seed=cfg.seed, source_config_hash=source["config_hash"],
                      initial_config_hash=initial["config_hash"], model_sha256=source["model_sha256"],
                      initial_model_sha256=initial["model_sha256"])
        row = measured.get(key)
        if row is None:
            records.append(dict(status="未実行: 保存済み重みなし", **common))
            continue
        for layer in row["layers"]:
            records.append(dict(status="実測", **common, config_hash=key,
                                mean_pixel_population_sd_across_samples=row["mean_pixel_population_sd_across_samples"],
                                **layer))
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in r.items()}
                         for r in records)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    planned = design(read_results(RUNS), runtime_info(device, filtered=True))
    prior = read_results(LEDGER)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    for cfg, source, initial in planned:
        key = digest(spec_for(source, initial))
        if key in prior or args.export_only:
            continue
        try:
            with FileLock(GPU_LOCK):
                row = measure(cfg, source, initial, device)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            print(f"未実行: {cfg.loss} seed={cfg.seed}; 保存済み重みなし: {exc}", flush=True)
            continue
        with FileLock(str(LEDGER) + ".lock"):
            existing = read_results(LEDGER, repair=True).get(key)
            if existing is not None:
                if existing != row:
                    raise ValueError("Conflicting decoder measurement for the same saved model")
            else:
                with LEDGER.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        prior[key] = row
        print(f"L {cfg.loss} seed={cfg.seed} mean pixel SD across samples="
              f"{row['mean_pixel_population_sd_across_samples']:.9g}; verified saved model", flush=True)
    export(planned, prior)
    print(f"L complete measurements {sum(digest(spec_for(s,i)) in prior for _,s,i in planned)}/10; "
          f"missing saved weights {len(missing)}. No training performed.", flush=True)


if __name__ == "__main__":
    main()
