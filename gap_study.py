"""Select the smallest reliable two-frame gap, then calibrate LR on separate data."""
import argparse
from dataclasses import asdict
import json
import os
import time

from filelock import FileLock
import numpy as np
import torch

from run_sweep import LOSSES, RATES, RUNS, digest, read_results
from training import Config, resolve_device, run_experiment

GAPS = (1, 2, 4, 8)
GAP_RADII = (2, 5, 8)
GAP_SEEDS = (0, 1, 2, 3, 4)
LR_SEEDS = (1729, 1730, 1731)
STEPS = 3000
GAP_LR = 0.003
VELOCITY_THRESHOLD = 0.8
R2_MARGIN = 0.01
SELECTION = "gap_selection"
CALIBRATION = "gap_calibration"


def runtime_info(device):
    return {"protocol": "circle-ae-gap-v1", "device": str(resolve_device(device)),
            "torch": str(torch.__version__), "numpy": np.__version__}


def selection_configs():
    return [Config(radius=r, frame_gap=gap, seed=seed, steps=STEPS, lr=GAP_LR)
            for gap in GAPS for r in GAP_RADII for seed in GAP_SEEDS]


def calibration_configs(gap):
    if gap not in GAPS:
        raise ValueError(f"Unsupported gap {gap}")
    return [Config(radius=5, frame_gap=gap, loss=loss, seed=seed, lr=lr, steps=STEPS)
            for loss in LOSSES for lr in RATES for seed in LR_SEEDS]


def study_spec(config, purpose, runtime):
    return {"config": asdict(config), "purpose": purpose, "runtime": runtime}


def _complete_rows(configs, purpose, rows, runtime):
    keys = [digest(study_spec(cfg, purpose, runtime)) for cfg in configs]
    missing = sum(k not in rows for k in keys)
    if missing:
        raise ValueError(f"{purpose}: {missing} of {len(keys)} measurements still missing")
    return [rows[k] for k in keys]


def selected_gap(rows, runtime):
    """Return None only after all 60 scores are present and no gap meets the rule."""
    measured = _complete_rows(selection_configs(), SELECTION, rows, runtime)
    for gap in GAPS:
        scores = [r["metrics"]["velocity_r2_mlp"] for r in measured
                  if r["config"]["radius"] in (5, 8) and r["config"]["frame_gap"] == gap]
        if all(score > VELOCITY_THRESHOLD for score in scores):
            return gap
    return None


def selected_rates(rows, runtime, gap):
    measured = _complete_rows(calibration_configs(gap), CALIBRATION, rows, runtime)
    rates = {}
    for loss in LOSSES:
        means = {}
        for lr in RATES:
            subset = [r["metrics"] for r in measured
                      if r["config"]["loss"] == loss and r["config"]["lr"] == lr]
            means[lr] = {name: float(np.mean([s[name] for s in subset]))
                         for name in ("velocity_r2_mlp", "position_r2_mlp", "psnr")}
        eligible = list(RATES)
        for name in ("velocity_r2_mlp", "position_r2_mlp"):
            best = max(means[lr][name] for lr in eligible)
            eligible = [lr for lr in eligible if means[lr][name] >= best - R2_MARGIN]
        rates[loss] = max(eligible, key=lambda lr: (means[lr]["psnr"], -lr))
    return rates


def _old_diagnostic(config, runtime, rows):
    """Only three original step-diagnostic fits match the entire gap-selection protocol."""
    if not (config.frame_gap == 1 and config.radius == 5 and config.seed in (0, 1, 2)):
        return None
    old_config = asdict(config)
    del old_config["frame_gap"]
    old_spec = {"config": old_config, "purpose": "step_diagnostic",
                "runtime": {**runtime, "protocol": "circle-ae-step-diagnostic-v1"}}
    source = rows.get(digest(old_spec))
    if source and "probe" in source and "temporal_reconstruction" in source:
        return source
    return None


def _run(config, purpose, runtime, rows):
    spec = study_spec(config, purpose, runtime)
    key = digest(spec)
    if key in rows:
        return
    splits = (2, 3) if purpose == SELECTION else (4, 5)
    source = _old_diagnostic(config, runtime, rows) if purpose == SELECTION else None
    print(f"{purpose} gap={config.frame_gap} r={config.radius} seed={config.seed} "
          f"{config.loss} lr={config.lr:g}", flush=True)
    if source is not None:
        # Recorded costs and predictions are from the source checkpoint. Reuse is explicit.
        outcome = {field: source[field] for field in ("device", "model_sha256", "last_train_loss",
                   "metrics", "temporal_reconstruction", "probe", "timing")}
        outcome["reused_from"] = source["config_hash"]
    elif purpose == SELECTION:
        outcome = run_experiment(config, runtime["device"], diagnostic=True, probe_splits=splits)
    else:
        outcome = run_experiment(config, runtime["device"], temporal=True, probe_splits=splits)
    row = {**spec, **outcome, "config_hash": key, "first_stage": None,
           "probe_fit_stream": [splits[0], 0], "probe_score_stream": [splits[1], 0]}
    packed = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with RUNS.open("ab") as out:
        out.write((packed + "\n").encode())
        out.flush()
        os.fsync(out.fileno())
    rows[key] = row
    print(f"saved {key[:10]} velocity R²={row['metrics']['velocity_r2_mlp']:.5f}"
          f"{' (reused)' if source is not None else ''}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--selection", action="store_true")
    phase.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runtime = runtime_info(args.device)
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUNS) + ".lock"):
        rows = read_results(RUNS, repair=not args.dry_run)
        if args.selection:
            purpose, configs = SELECTION, selection_configs()
        else:
            gap = selected_gap(rows, runtime)
            if gap is None:
                print("No tested gap met the reliability criterion. LR selection has no target.")
                return
            purpose, configs = CALIBRATION, calibration_configs(gap)
        missing = sum(digest(study_spec(cfg, purpose, runtime)) not in rows for cfg in configs)
        print(f"{purpose}: {len(configs)} configurations, {missing} outstanding", flush=True)
        if args.dry_run:
            return
        start = time.monotonic()
        for config in configs:
            _run(config, purpose, runtime, rows)
        if args.selection:
            print(f"selected gap: {selected_gap(rows, runtime)}")
        else:
            print(f"selected learning rates: {selected_rates(rows, runtime, gap)}")
        print(f"elapsed {time.monotonic() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
