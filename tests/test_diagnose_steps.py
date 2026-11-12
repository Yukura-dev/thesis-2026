"""A checkpoint must reproduce uninterrupted Adam and data-stream updates."""
from dataclasses import replace

import diagnose_steps as diagnosis
from training import Config, resolve_device, run_experiment


def test_resumed_checkpoint_matches_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnosis, "ROOT", tmp_path)
    monkeypatch.setattr(diagnosis, "BUDGETS", (2, 4))
    device = resolve_device("cpu")
    runtime = diagnosis.runtime_info(device)
    cfg = Config(radius=5, lr=0.003, batch_size=4, steps=4, seed=1)
    diagnosis.train_to(replace(cfg, steps=2), device, runtime)
    diagnosis.train_to(cfg, device, runtime)
    final_path = diagnosis.checkpoint_path(runtime, cfg.seed, 4)
    state = diagnosis.load_checkpoint(final_path, runtime, cfg)
    # Replicate with a fresh optimizer and uninterrupted stream, at the default
    # run's normal entry point. Evaluation does not contribute to the hash.
    uninterrupted = run_experiment(cfg, device, calibration=True)
    assert state["model_sha256"] == uninterrupted["model_sha256"]
    assert state["last_train_loss"] == uninterrupted["last_train_loss"]
