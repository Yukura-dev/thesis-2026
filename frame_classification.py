"""J: classify both reconstructed frames using only existing saved model weights.

Flat backgrounds use mean center intensity >= 0.65. Textured backgrounds use
mean(predicted center - preoverlay background) >= mean(foreground - background)/2.
The original rows and the historical raw-center H rule are never overwritten.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
import time

from filelock import FileLock
import numpy as np
import torch

from data import MovieStream, SIZE, smooth_backgrounds
from evaluation import AffineMoments, ImageMetrics
from run_sweep import ROOT, RUNS, config_record, digest, identity, read_results, runtime_info as main_runtime, stage3_arms
from training import Autoencoder, Config, _configure, _parameter_digest, resolve_device

PROTOCOL = "circle-ae-frame-classification-v1"
PURPOSE = "frame_classification_measurement"
LEDGER = ROOT / "results" / "frame_classification_measurements.jsonl"
GPU_LOCK = ROOT / "checkpoints" / "final_jkl_gpu.lock"
CLASSES = ("background_only", "past_only", "current_only", "both")
FRAME_NAMES = ("previous", "current")
RULE = "mean(predicted_center-preoverlay_background) >= mean(foreground-preoverlay_background)/2"


def classify_frames(previous, current, threshold=.65):
    if not all(math.isfinite(x) for x in (previous, current, threshold)):
        raise ValueError("Classification needs finite, unrounded frame means")
    return CLASSES[int(previous >= threshold) + 2 * int(current >= threshold)]


def borderline_frames(previous, current, threshold=.65, margin=.05):
    if not all(math.isfinite(x) for x in (previous, current, threshold, margin)) or margin < 0:
        raise ValueError("Borderline classification needs finite values and a nonnegative margin")
    return tuple(abs(x - threshold) <= margin for x in (previous, current))


def classify_measurement(metrics, foreground=1., textured=False):
    prediction = [metrics[f"center_intensity_{name}"] for name in FRAME_NAMES]
    background = [metrics[f"preoverlay_background_center_{name}"] for name in FRAME_NAMES]
    # Use the requested exact flat threshold, avoiding float32(0.3) roundoff.
    margins = [(p - b) - (foreground - b)/2 for p, b in zip(prediction, background)] if textured else [p - .65 for p in prediction]
    result = {"classification": classify_frames(*margins, threshold=0.),
              "raw_comparison_classification": classify_frames(*prediction),
              "classification_rule": RULE if textured else "mean(predicted_center) >= 0.65",
              "borderline_rule": "abs(signed classification margin) <= 0.05"}
    result.update({f"signed_margin_{name}": value for name, value in zip(FRAME_NAMES, margins)})
    result.update({f"borderline_{name}": abs(value) <= .05 for name, value in zip(FRAME_NAMES, margins)})
    return result


def analysis_loss(source):
    return "l2_to_l1" if source.get("method") == "l2_3000_l1_3000" else source["config"]["loss"]


def model_identity(source):
    return digest({"config": config_record(Config(**source["config"])), "model_sha256": source["model_sha256"]})


def design(rows, device="mps"):
    """514 unique models: A/B/D + D controls, 120 G endpoints, 40 H textures.

    Group membership may overlap; a model is measured and counted globally once.
    G's forty intermediate anchors are not extra members of this population.
    """
    from switch_study import plan, config_for, endpoint_spec, runtime_info as g_runtime
    from texture_study import configs as h_configs, experiment_spec as h_spec, runtime_info as h_runtime
    rt = main_runtime(device, filtered=True)
    base = Config(frame_gap=4, steps=3000, reject_reflections=True)
    arms = stage3_arms(base, {"l1": .003, "l2": .003})
    selected = {}

    def add(spec, arm, role="primary"):
        source_key = digest(spec)
        if source_key not in rows:
            raise ValueError(f"未実行: source result missing: {source_key} ({arm}/{role})")
        source = rows[source_key]
        if source.get("probe_splits") != [6, 7] or source["config"]["probe_test"] != 5000:
            raise ValueError(f"Source must use the fixed 5,000-example split 7: {source_key}")
        key = model_identity(source)
        target = selected.setdefault(key, {"source": source, "model_identity": key,
                                           "memberships": [], "source_aliases": []})
        member = {"arm": arm, "role": role}
        if member not in target["memberships"]:
            target["memberships"].append(member)
        if source_key not in target["source_aliases"]:
            target["source_aliases"].append(source_key)
        if source["metrics"] != target["source"]["metrics"]:
            raise ValueError("Duplicate model/config identities have inconsistent recorded metrics")

    for arm in ("A", "B", "D"):
        for cfg in arms[arm]:
            add(identity(cfg, "experiment", rt), arm)
    for cfg in dict.fromkeys(replace(cfg, placement="target_only") for cfg in arms["D"]):
        add(identity(cfg, "experiment", rt), "D", "target_only_control")
    for radius, level, seed, method in plan():
        add(endpoint_spec(config_for(radius, level, seed), method, g_runtime(device)), "G")
    for cfg in h_configs():
        add(h_spec(cfg, h_runtime(device)), "H")
    for radius in (5, 8):
        for loss in ("l1", "l2"):
            for seed in range(5):
                cfg = replace(base, radius=radius, loss=loss, seed=seed, lr=.003)
                add(identity(cfg, "experiment", rt), "H", "flat_control")
    return list(selected.values())


def saved_checkpoint(source):
    """Return a saved file only. Never import a helper that can retrain a model."""
    key = source["config_hash"]
    if source["purpose"] == "final_g_switch":
        candidates = [ROOT / "checkpoints" / "final_g" / key / "6000.pt"]
    elif source["purpose"] == "background_texture":
        candidates = [ROOT / "checkpoints" / "background_texture" / key / "weights.pt"]
    elif source["purpose"] == "experiment":
        candidates = [ROOT / "checkpoints" / folder / key / "weights.pt"
                      for folder in ("stage3", "collapse_replay")]
    else:
        raise ValueError(f"Unsupported saved-only source purpose {source['purpose']}")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"未実行: saved weights missing for {key}; tried {candidates}")


def load_saved(source, device):
    path = saved_checkpoint(source)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "spec" in state:
        expected = {name: source[name] for name in ("config", "purpose", "runtime")}
        if state["spec"] != expected or state["step"] != source["config"]["steps"]:
            raise ValueError(f"Saved G identity/budget differs: {path}")
    elif state.get("source_config_hash") != source["config_hash"]:
        raise ValueError(f"Saved source identity differs: {path}")
    cfg = Config(**source["config"])
    _configure(cfg.seed)
    model = Autoencoder(cfg.latent_dim, cfg.background).to(device)
    model.load_state_dict(state["model"])
    if _parameter_digest(model) != source["model_sha256"]:
        raise ArithmeticError(f"Saved model SHA differs from original result: {path}")
    model.eval().requires_grad_(False)
    return model, cfg, path


def sample_centers(images, centers):
    """The same bilinear interpolation and float32 center coordinates as ImageMetrics."""
    image = images.reshape(-1, SIZE, SIZE)
    centers = np.asarray(centers).reshape(-1, 2)
    lower = np.floor(centers).astype(np.int64)
    dx, dy = (centers - lower).T
    x, y = lower.T
    k = np.arange(len(image))
    values = ((1-dy)*((1-dx)*image[k, y, x] + dx*image[k, y, x+1]) +
              dy*((1-dx)*image[k, y+1, x] + dx*image[k, y+1, x+1]))
    return values.reshape(-1, 2)


def sample_with_background(stream, count):
    # Clone before sample(), so drawing the preoverlay texture never advances the
    # actual data stream or changes a movie, a label, or the independent noise.
    if stream.background_texture_sd:
        rng = np.random.default_rng(0)
        rng.bit_generator.state = deepcopy(stream.texture_rng.bit_generator.state)
        background = smooth_backgrounds(rng, count, stream.background, stream.background_texture_sd)
    else:
        background = np.full((count, SIZE, SIZE), stream.background, dtype=np.float32)
    batch = stream.sample(count, need_target=False)
    background_pair = np.broadcast_to(background[:, None], batch.clean.shape)
    return batch, background_pair


def measurement_spec(source, device):
    return {"config": source["config"], "purpose": PURPOSE,
            "runtime": {**main_runtime(device, filtered=True), "protocol": PROTOCOL,
                        "source_config_hash": source["config_hash"], "source_model_sha256": source["model_sha256"],
                        "classification_rule": RULE, "borderline_margin": .05}}


def measure(source, device):
    started = time.perf_counter()
    model, cfg, path = load_saved(source, device)
    stats = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
    has_affine = "corrected_psnr" in source["metrics"]
    moments = AffineMoments() if has_affine else None
    background_sum = np.zeros(2, dtype=np.float64)
    clean_sum = np.zeros(2, dtype=np.float64)
    stream = MovieStream(np.random.SeedSequence([7, 0]), **cfg.movie_kwargs())
    with torch.inference_mode():
        for start in range(0, cfg.probe_test, cfg.batch_size):
            batch, background = sample_with_background(stream, min(cfg.batch_size, cfg.probe_test-start))
            pred = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
            stats.update(pred, batch, cfg.radius)
            background_sum += sample_centers(background, batch.centers).sum(axis=0, dtype=np.float64)
            clean_sum += sample_centers(batch.clean, batch.centers).sum(axis=0, dtype=np.float64)
            if moments is not None:
                moments.update(pred, batch.clean)
        measured = stats.result()
        if moments is not None:
            a, b = moments.coefficients()
            sse = 0.
            stream = MovieStream(np.random.SeedSequence([7, 0]), **cfg.movie_kwargs())
            for start in range(0, cfg.probe_test, cfg.batch_size):
                batch = stream.sample(min(cfg.batch_size, cfg.probe_test-start), need_target=False)
                pred = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
                sse += np.square(a*pred.astype(np.float64)+b-batch.clean).sum()
            measured.update(corrected_psnr=-10*math.log10(sse/moments.n), affine_a=a, affine_b=b,
                            affine_n_pixels=moments.n, output_pixel_sd=math.sqrt(moments.sxx/moments.n))
    differences = {name: abs(value-source["metrics"][name]) for name, value in measured.items()
                   if name in source["metrics"]}
    if any(difference != 0 for difference in differences.values()):
        raise ArithmeticError(f"Original image metrics are not exactly reproduced for {source['config_hash']}: {differences}")
    for index, name in enumerate(FRAME_NAMES):
        bg = float(background_sum[index]/cfg.probe_test)
        measured[f"preoverlay_background_center_{name}"] = bg
        measured[f"clean_center_intensity_{name}"] = float(clean_sum[index]/cfg.probe_test)
        measured[f"foreground_minus_background_{name}"] = cfg.foreground-bg
        measured[f"reconstructed_minus_background_{name}"] = measured[f"center_intensity_{name}"]-bg
    spec = measurement_spec(source, device)
    return {**spec, "config_hash": digest(spec), "source_config_hash": source["config_hash"],
            "model_identity": model_identity(source), "model_sha256": source["model_sha256"],
            "checkpoint": str(path.relative_to(ROOT)), "weights_only": True,
            "probe_score_stream": [7, 0], "evaluation_pairs": cfg.probe_test,
            "original_image_metrics_exact": True, "original_image_metric_absolute_differences": differences,
            "metrics": measured, **classify_measurement(measured, cfg.foreground, bool(cfg.background_texture_sd)),
            "evaluation_seconds": time.perf_counter()-started}


def append(row):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LEDGER)+".lock"):
        existing = read_results(LEDGER, repair=True)
        if row["config_hash"] in existing:
            return existing[row["config_hash"]]
        with LEDGER.open("a") as out:
            out.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))+"\n")
            out.flush()
            os.fsync(out.fileno())
    return row


def audit(targets):
    paths, missing = [], []
    for target in targets:
        try:
            paths.append(saved_checkpoint(target["source"]))
        except FileNotFoundError as exc:
            missing.append(str(exc))
    memberships = Counter(member["arm"] for target in targets for member in target["memberships"])
    return {"unique_models": len(targets), "memberships": dict(memberships),
            "saved_checkpoints": len(paths), "checkpoint_folders": dict(Counter(path.parts[-3] for path in paths)),
            "missing": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    targets = design(read_results(), str(device))
    check = audit(targets)
    prior = read_results(LEDGER)
    remaining = sum(digest(measurement_spec(t["source"], device)) not in prior for t in targets)
    print(json.dumps({**check, "remaining_measurements": remaining}, ensure_ascii=False), flush=True)
    if check["missing"]:
        raise FileNotFoundError("未実行: saved checkpoints are missing; no retraining fallback is allowed")
    if args.plan:
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets, 1):
        source = target["source"]
        key = digest(measurement_spec(source, device))
        if key not in prior:
            with FileLock(str(GPU_LOCK)):
                current = read_results(LEDGER, repair=True)
                row = current.get(key)
                if row is None:
                    row = append(measure(source, device))
            time.sleep(.15)
            prior[key] = row
            print(f"J {index}/{len(targets)} {source['config_hash'][:10]} {row['classification']} "
                  f"centers={row['metrics']['center_intensity_previous']:.5f}/{row['metrics']['center_intensity_current']:.5f}", flush=True)
    print(f"J complete: {len(targets)} unique saved models, no retraining", flush=True)


if __name__ == "__main__":
    main()
