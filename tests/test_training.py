import json
import os
from dataclasses import asdict

import pytest

from training import Config, run_experiment


@pytest.fixture(scope="module")
def large_clean():
    """Share only the first fit: the independent repeat always starts from scratch."""
    cfg = Config(radius=8, loss="l2", lr=float(os.environ.get("AE_SANITY_LR", "0.003")))
    device = os.environ.get("AE_DEVICE", "auto")
    result = run_experiment(cfg, device)
    print("sanity_fit", json.dumps({"config": asdict(cfg), **result}, sort_keys=True))
    return cfg, device, result


def test_large_clean_circle(large_clean):
    score = large_clean[2]["metrics"]["position_r2_mlp"]
    assert score > 0.95, f"Held-out position MLP R²={score:.6f} (required > 0.95)"


def test_repeatable_training(large_clean):
    """Same full 300-step fit, including fresh initialization and fresh 10k/5k probes."""
    cfg, device, first = large_clean
    second = run_experiment(cfg, device)
    for key in ("model_sha256", "validation_loss", "last_train_loss", "metrics"):
        assert first[key] == second[key], (key, first[key], second[key])
    print("Repeat fit verified:", second["model_sha256"])
