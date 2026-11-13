"""Select a gap after rejecting observed-window wall hits and validate learning rates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import os

from filelock import FileLock
import numpy as np

import gap_study as original
from run_sweep import RUNS, digest, read_results
from training import run_experiment

GAPS = (2, 4)
RADII = (2, 5, 8)
SELECTION = "filtered_gap_selection"
CALIBRATION = "filtered_gap_calibration"
ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / "results" / "filtered_geometry.json"
ORIGINAL_ORACLE = ROOT / "results" / "reflection_oracle.json"
ORACLE_THRESHOLD = 0.95


def runtime_info(device):
    return {**original.runtime_info(device), "protocol": "circle-ae-filtered-gap-v1"}


def _centers(path=ORACLE):
    evidence = json.loads(Path(path).read_text())
    if (evidence["protocol"] != "observed-no-reflection-v1" or
            evidence["reject_reflections"] is not True or
            evidence["fit"] != {"n": 10000, "seed_sequence": [2, 0]} or
            evidence["test"] != {"n": 5000, "seed_sequence": [3, 0]}):
        raise ValueError("Filtered oracle and AE probe evidence use different protocols")
    centers = {(r["frame_gap"], r["radius"]): r["oracle"]["mlp"]["r2"]
               for r in evidence["conditions"] if r["frame_gap"] in GAPS}
    if set(centers) != {(g, r) for g in GAPS for r in RADII} or not all(np.isfinite(list(centers.values()))):
        raise ValueError("Six finite filtered true-center MLP scores are required")
    return centers


def oracle_passed(path=ORACLE):
    return all(v >= ORACLE_THRESHOLD for v in _centers(path).values())


def original_gate_reached():
    prior = json.loads(ORIGINAL_ORACLE.read_text())
    subset = {(row["frame_gap"], row["radius"]): row["oracle_mlp"]["all"]["r2"]
              for row in prior["conditions"] if row["frame_gap"] in GAPS and row["radius"] in (5, 8)}
    if set(subset) != {(g, r) for g in GAPS for r in (5, 8)}:
        raise ValueError("Missing unfiltered true-center MLP measurements")
    return any(v < ORACLE_THRESHOLD for v in subset.values())


def selection_configs():
    return [replace(c, reject_reflections=True) for c in original.selection_configs() if c.frame_gap in GAPS]


def calibration_configs(gap):
    if gap not in GAPS:
        raise ValueError(f"Not a filtered-study candidate gap: {gap}")
    return [replace(c, reject_reflections=True) for c in original.calibration_configs(gap)]


def selected_gap(rows, runtime):
    measured = original._complete_rows(selection_configs(), SELECTION, rows, runtime)
    for gap in GAPS:
        if all(row["metrics"]["velocity_r2_mlp"] > original.VELOCITY_THRESHOLD for row in measured
               if row["config"]["frame_gap"] == gap and row["config"]["radius"] in (5, 8)):
            return gap
    return None


def selected_rates(rows, runtime, gap):
    # The same three-training-seed tie break applies before and after filtering.
    measured = original._complete_rows(calibration_configs(gap), CALIBRATION, rows, runtime)
    return original.rates_from_measurements(measured)


def _run(config, purpose, runtime, rows, centers):
    spec = original.study_spec(config, purpose, runtime)
    key = digest(spec)
    if key in rows:
        return
    print(f"{purpose} Δ={config.frame_gap} r={config.radius} seed={config.seed} "
          f"{config.loss} lr={config.lr:g}", flush=True)
    splits = (2, 3) if purpose == SELECTION else (4, 5)
    outcome = run_experiment(config, runtime["device"], diagnostic=(purpose == SELECTION),
                             temporal=(purpose == CALIBRATION), probe_splits=splits)
    row = {**spec, **outcome, "config_hash": key, "first_stage": None,
           "probe_fit_stream": [splits[0], 0], "probe_score_stream": [splits[1], 0]}
    if purpose == SELECTION:
        row["true_center_velocity_r2_mlp"] = centers[config.frame_gap, config.radius]
        row["true_center_source"] = "results/filtered_geometry.json"
    line = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with RUNS.open("ab") as out:
        out.write((line + "\n").encode())
        out.flush()
        os.fsync(out.fileno())
    rows[key] = row
    print(f"saved {key[:10]}: velocity MLP R² {row['metrics']['velocity_r2_mlp']:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--selection", action="store_true")
    phase.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not original_gate_reached():
        print("Unfiltered oracles all reached the threshold. Filtered selection has no indication.")
        return
    centers = _centers()
    runtime = runtime_info(args.device)
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUNS) + ".lock"):
        rows = read_results(RUNS, repair=not args.dry_run)
        if args.selection:
            purpose, configs = SELECTION, selection_configs()
        else:
            gap = selected_gap(rows, runtime)
            if gap is None or not oracle_passed():
                print(f"No LR calibration: selected_gap={gap}, six_oracles_passed={oracle_passed()}")
                return
            purpose, configs = CALIBRATION, calibration_configs(gap)
        missing = sum(digest(original.study_spec(cfg, purpose, runtime)) not in rows for cfg in configs)
        print(f"{purpose}: {len(configs)} configurations, {missing} outstanding", flush=True)
        if args.dry_run:
            return
        for cfg in configs:
            _run(cfg, purpose, runtime, rows, centers)
        if args.selection:
            print(f"filtered selected gap: {selected_gap(rows, runtime)}")
        else:
            print(f"filtered selected LRs: {selected_rates(rows, runtime, gap)}")


if __name__ == "__main__":
    main()
