"""Evaluate saved stage 1/2 weights (or bitwise-verified replays) on their original images."""
import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

from filelock import FileLock
import numpy as np
import torch

from data import MovieStream
from evaluation import AffineMoments, ImageMetrics
from reflection_study import weights_for
from run_sweep import (ROOT, RUNS, chosen_gap, digest, identity, main_rates, read_results,
                       runtime_info, stage_configs)
from training import Config, _validate, resolve_device

LEDGER = ROOT / "results" / "collapse_reconstruction.jsonl"
CSV_PATH = ROOT / "results" / "collapse_reconstruction_seeds.csv"
FIGURE = ROOT / "figures" / "collapse_reconstructions.png"
CACHE = ROOT / "checkpoints" / "collapse_replay"
GPU_LOCK = ROOT / "checkpoints" / "collapse_gpu.lock"
EXAMPLES = (("gaussian", 0.), ("salt_pepper", .2), ("gaussian", .1), ("gaussian", .4))
PROTOCOL = "circle-ae-collapse-image-v1"
FIELDS = ("source_config_hash", "model_sha256", "comparison", "radius", "loss", "lr", "noise_type",
          "noise_level", "seed", "psnr", "corrected_psnr", "affine_a", "affine_b", "center_intensity",
          "center_contrast", "object_mae", "background_mae", "output_pixel_sd", "mean_frame_pixel_sd",
          "background_psnr", "validation_l1")


def db(mse):
    return -10 * math.log10(mse) if mse > 0 else math.inf


def design(device):
    original = read_results(RUNS)
    rt = runtime_info(device, filtered=True)
    gap = chosen_gap(original, rt)
    base = Config(frame_gap=gap, steps=3000, reject_reflections=True)
    rates = main_rates(base, rt, original)
    groups = (("stage1", stage_configs(1, base, rates)), ("stage2", stage_configs(2, base, rates)),
              ("stage1_shared", stage_configs(1, base, rates, shared_lr=True)))
    designs = {}
    for name, configs in groups:
        for cfg in configs:
            key = digest(identity(cfg, "experiment", rt))
            if key not in original:
                raise ValueError(f"Missing historical fit {key}; this diagnostic requires completed main runs")
            src = original[key]
            if src["probe_splits"] != [6, 7] or src["first_stage"] not in (1, 2):
                raise ValueError(f"Wrong evaluation split/stage for {key}")
            if key not in designs:
                designs[key] = (cfg, src, [])
            designs[key][2].append(name)
    return list(designs.values())


def spec_for(source):
    runtime = {**source["runtime"], "protocol": PROTOCOL, "source_config_hash": source["config_hash"]}
    return {"config": source["config"], "runtime": runtime, "purpose": "collapse_reconstruction"}


def is_shared_anchor(cfg):
    return (cfg.lr == .003 and cfg.noise_level == 0 and
            (cfg.radius, cfg.loss) in ((2, "l2"), (5, "l2"), (5, "l1"), (8, "l1")))


def is_example(cfg, groups):
    return (cfg.seed == 0 and cfg.radius == 5 and cfg.loss in ("l1", "l2") and
            any(g in ("stage1", "stage2") for g in groups) and (cfg.noise_type, cfg.noise_level) in EXAMPLES)


def prioritize(trial):
    cfg, source, groups = trial
    cached = (CACHE / source["config_hash"] / "weights.pt").exists()
    # Share the 20 clean continuation anchors with collapse_trajectories when both jobs are run.
    deferred = is_shared_anchor(cfg) and not cached
    p45 = cfg.radius == 5 and cfg.noise_type == "salt_pepper" and cfg.noise_level == .45
    return deferred, not p45, not is_example(cfg, groups), "stage1_shared" in groups


@torch.inference_mode()
def test_batches(model, cfg, device):
    stream = MovieStream(np.random.SeedSequence([7, 0]), **cfg.movie_kwargs())
    for start in range(0, cfg.probe_test, cfg.batch_size):
        batch = stream.sample(min(cfg.batch_size, cfg.probe_test - start), need_target=False)
        pred = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
        yield batch, pred


def save_example(path, inputs, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as file:
        np.savez_compressed(file, inputs=inputs, output=output)
    tmp.replace(path)


def image_readouts(cfg, model, source, device, sample_path=None):
    model.eval()
    original_metrics = ImageMetrics(include_center_intensity=True)
    moment = AffineMoments()
    baseline_sse = frame_sd = 0.
    for index, (batch, pred) in enumerate(test_batches(model, cfg, device)):
        original_metrics.update(pred, batch, cfg.radius)
        moment.update(pred, batch.clean)
        baseline_sse += np.square(batch.clean.astype(np.float64) - np.float32(cfg.background)).sum()
        frame_sd += np.std(pred, axis=(-2, -1), ddof=0, dtype=np.float64).sum()
        if index == 0 and sample_path is not None:
            save_example(sample_path, batch.input[0], pred[0])
    historical = original_metrics.result()
    if any(v != source["metrics"][k] for k, v in historical.items()):
        raise ArithmeticError(f"Original reconstruction metrics do not match for {source['config_hash']}")
    expected_validation = _validate(model, cfg, device)
    if expected_validation != source["validation_loss"]:
        raise ArithmeticError(f"Original validation score does not match for {source['config_hash']}")
    # An independent second pass evaluates actual squared residuals, avoiding cancellation
    # in syy - sxy**2/sxx when slope or reconstruction fidelity is high. Never clip.
    a, b = moment.coefficients()
    adjusted_sse = 0.
    for batch, pred in test_batches(model, cfg, device):
        residual = a * pred.astype(np.float64) + b - batch.clean
        adjusted_sse += np.square(residual).sum()
    corrected = db(adjusted_sse / moment.n)
    if corrected + 1e-6 < historical["psnr"]:
        raise ArithmeticError("Fitted affine transform increased the held-out pooled MSE")
    val_l1 = None
    if cfg.radius == 5 and cfg.noise_level == 0 and cfg.lr == .003 and cfg.loss in ("l1", "l2"):
        val_l1 = _validate(model, replace(cfg, loss="l1"), device)
    return {**historical, "corrected_psnr": corrected, "affine_a": a, "affine_b": b,
            "output_pixel_sd": math.sqrt(moment.sxx / moment.n),
            "mean_frame_pixel_sd": float(frame_sd / (2 * cfg.probe_test)),
            "background_psnr": db(baseline_sse / moment.n), "validation_l1": val_l1,
            "n_pixels": moment.n, "n_test_pairs": cfg.probe_test}


def measure(trial, device):
    cfg, source, groups = trial
    spec = spec_for(source)
    key = digest(spec)
    cache_dir = CACHE / source["config_hash"]
    # The shared helper recognises unfiltered delta=1 legacy files. This study uses
    # the selected delta=4 clean-interval distribution and only full matching hashes.
    if cfg.frame_gap != 4 or not cfg.reject_reflections:
        raise ValueError("This archived replay is defined for the selected filtered delta=4 fits")
    with FileLock(str(GPU_LOCK)):
        model, provenance = weights_for(cfg, source, device, cache_dir / "weights.pt")
        metrics = image_readouts(cfg, model, source, device,
                                 cache_dir / "example.npz" if is_example(cfg, groups) else None)
    record = {**spec, "config_hash": key, "source_config_hash": source["config_hash"],
              "model_sha256": source["model_sha256"], "weight_provenance": provenance,
              "original_validation_exact": True, "original_reconstruction_metrics_exact": True,
              "comparison": "+".join(groups), "probe_score_stream": [7, 0], "metrics": metrics}
    line = json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with LEDGER.open("ab") as file:
        file.write((line + "\n").encode())
        file.flush()
        os.fsync(file.fileno())
    return record


def export_table(trials, rows):
    records = [rows[digest(spec_for(source))] for _, source, _ in trials]
    tmp = CSV_PATH.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({**r, **r["config"], **r["metrics"]})
    tmp.replace(CSV_PATH)
    return records


def draw_examples(trials, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = {}
    for cfg, source, groups in trials:
        if is_example(cfg, groups):
            cache_dir = CACHE / source["config_hash"]
            path = cache_dir / "example.npz"
            if not path.exists():
                # Git excludes the weight/prediction cache. Regenerate seed-0 examples
                # at exactly the original source hashes when running on a fresh checkout.
                with FileLock(str(GPU_LOCK)):
                    model, _ = weights_for(cfg, source, device, cache_dir / "weights.pt")
                    model.eval()
                    batch, pred = next(test_batches(model, cfg, device))
                save_example(path, batch.input[0], pred[0])
            with np.load(path) as img:
                images[cfg.noise_type, cfg.noise_level, cfg.loss] = (img["inputs"], img["output"])
    fig, axes = plt.subplots(8, 3, figsize=(6.6, 11.7))
    fig.subplots_adjust(left=.23, right=.93, bottom=.054, top=.93, hspace=.11, wspace=.09)
    fig.suptitle("Reconstruction of the same observation", y=.986, fontsize=13.5)
    fig.text(.5, .956, "r = 5 px   ·   Δ = 4   ·   AE seed 0   ·   evaluation pair 0", ha="center", fontsize=9)
    for j, title in enumerate(("Input", "L1", "L2")):
        axes[0, j].set_title(title, fontsize=11)
    labels = ("Clean", "Salt–pepper\np = 0.20", "Gaussian\nσ = 0.10", "Gaussian\nσ = 0.40")
    for i, setting in enumerate(EXAMPLES):
        inputs, l1 = images[*setting, "l1"]
        other_inputs, l2 = images[*setting, "l2"]
        if not np.array_equal(inputs, other_inputs) or not np.array_equal(inputs, images["gaussian", 0., "l1"][0]):
            raise ValueError("Example order mismatched between target-only training conditions")
        for ch in (0, 1):
            row = i*2 + ch
            for j, img in enumerate((inputs, l1, l2)):
                ax = axes[row, j]
                im = ax.imshow(img[ch], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[row, 0].set_ylabel("t − 4" if ch == 0 else "t", rotation=0, ha="right", va="center",
                                   labelpad=8, fontsize=8)
        box0, box1 = axes[2*i, 0].get_position(), axes[2*i+1, 0].get_position()
        fig.text(.022, (box0.y0 + box1.y1)/2, labels[i], va="center", fontsize=9)
    cax = fig.add_axes([.31, .022, .5, .008])
    fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .3, 1])
    cax.tick_params(labelsize=7)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = ap.parse_args()
    device = resolve_device(args.device)
    trials = design(device)
    GPU_LOCK.parent.mkdir(exist_ok=True, parents=True)
    with FileLock(str(LEDGER) + ".lock"):
        rows = read_results(LEDGER, repair=True)
        jobs = [t for t in trials if digest(spec_for(t[1])) not in rows]
        print(f"{len(trials)} original fits, {len(jobs)} reconstructions outstanding", flush=True)
        while jobs:
            # Pick newly available shared anchors first; this also releases example pairs quickly.
            jobs.sort(key=lambda t: (not (CACHE / t[1]["config_hash"] / "weights.pt").exists(), prioritize(t)))
            trial = jobs.pop(0)
            cfg = trial[0]
            print(f"r={cfg.radius} {cfg.loss} lr={cfg.lr} {cfg.noise_type}={cfg.noise_level} seed={cfg.seed}", flush=True)
            row = measure(trial, device)
            rows[row["config_hash"]] = row
            print(f"saved {row['source_config_hash'][:12]}: PSNR {row['metrics']['psnr']:.6f}, "
                  f"corrected {row['metrics']['corrected_psnr']:.6f}, sd {row['metrics']['output_pixel_sd']:.6g}", flush=True)
            time.sleep(.12)
        export_table(trials, rows)
        draw_examples(trials, device)
        print(f"Saved {CSV_PATH} and {FIGURE}", flush=True)


if __name__ == "__main__":
    main()
