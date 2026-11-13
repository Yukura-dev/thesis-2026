import json
import os
from dataclasses import asdict

import pytest

from run_sweep import SWEEP_STEPS, read_results
from training import Config, resolve_device, run_experiment


@pytest.fixture(scope="module")
def large_clean():
    """Share only the first fit: the independent repeat always starts from scratch."""
    cfg = Config(radius=8, loss="l2", steps=int(os.environ.get("AE_SANITY_STEPS", SWEEP_STEPS)),
                 lr=float(os.environ.get("AE_SANITY_LR", "0.003")))
    device = os.environ.get("AE_DEVICE", "auto")
    result = run_experiment(cfg, device)
    print("sanity_fit", json.dumps({"config": asdict(cfg), **result}, sort_keys=True))
    return cfg, device, result


def test_large_clean_circle(large_clean):
    score = large_clean[2]["metrics"]["position_r2_mlp"]
    assert score > 0.95, f"Held-out position MLP R²={score:.6f} (required > 0.95)"


def test_repeatable_training(large_clean):
    """Repeat the full fit with fresh initialization and fresh 10k/5k probes."""
    cfg, device, first = large_clean
    second = run_experiment(cfg, device)
    for key in ("model_sha256", "validation_loss", "last_train_loss", "metrics"):
        assert first[key] == second[key], (key, first[key], second[key])
    print("Repeat fit verified:", second["model_sha256"])


def test_large_clean_velocity():
    """Fifth check: use the qualified gap and probe-selected L2 rate on fresh data."""
    from gap_study import runtime_info, selected_gap, selected_rates
    device = resolve_device(os.environ.get("AE_DEVICE", "auto"))
    runtime = runtime_info(device)
    rows = read_results()
    try:
        gap = selected_gap(rows, runtime)
        if gap is None:
            pytest.skip("未実行: no frame gap met the selection criteria")
        rates = selected_rates(rows, runtime, gap)
    except ValueError as exc:
        pytest.skip(f"未実行: {exc}")
    cfg = Config(radius=8, loss="l2", steps=SWEEP_STEPS, frame_gap=gap, lr=rates["l2"], seed=0)
    result = run_experiment(cfg, device, probe_splits=(6, 7))
    print("sanity_gap_fit", json.dumps({"config": asdict(cfg), **result}, sort_keys=True))
    score = result["metrics"]["velocity_r2_mlp"]
    assert score > 0.8, f"Held-out velocity MLP R²={score:.6f} (required > 0.8)"


def test_filtered_large_clean_velocity():
    """The new fifth gate is fitted once and checked against the persisted selector decisions."""
    from filelock import FileLock
    from run_sweep import (RUNS, SANITY_PURPOSE, chosen_gap, filtered_sanity_config, main_rates,
                           require_passed_sanity, run_one, runtime_info)
    device = resolve_device(os.environ.get("AE_DEVICE", "auto"))
    runtime = runtime_info(device, filtered=True)
    rows = read_results()
    try:
        gap = chosen_gap(rows, runtime)
        rates = main_rates(Config(frame_gap=gap, reject_reflections=True), runtime, rows)
    except (ValueError, FileNotFoundError) as exc:
        pytest.skip(f"未実行: {exc}")
    cfg = filtered_sanity_config(rates, gap)
    with FileLock(str(RUNS) + ".lock"):
        rows = read_results(RUNS, repair=True)
        run_one(cfg, SANITY_PURPOSE, runtime, None, rows, RUNS)
    # Check persisted scores even on resume. No retraining to search for a passing seed.
    scored = require_passed_sanity(rows, cfg, runtime)
    print("filtered r=8 velocity R²", scored["metrics"]["velocity_r2_mlp"])
