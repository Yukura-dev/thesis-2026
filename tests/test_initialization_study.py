"""Frozen pre-K references prove compatibility; small fits test exact K recovery."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import initialization_study as k
from data import MovieStream
from diagnose_steps import build_model
from run_sweep import digest, identity, read_results
from training import Config, _configure, _parameter_digest, reconstruction_loss

REFERENCE = json.loads((Path(__file__).parent / "fixtures/initialization_legacy.json").read_text())


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def test_default_initial_parameters_keep_all_five_historical_shas():
    assert str(torch.__version__) == REFERENCE["torch"]
    for seed in range(5):
        cfg = Config(seed=seed, frame_gap=4, reject_reflections=True)
        model = build_model(cfg, torch.device("cpu"))
        assert _parameter_digest(model) == REFERENCE["initial"][str(seed)]
        explicit = k.build_model(cfg, "default", torch.device("cpu"))
        assert _parameter_digest(explicit) == _parameter_digest(model)


@pytest.mark.parametrize("loss", ("l1", "l2"))
def test_default_training_and_held_out_results_match_prechange_fixture(loss):
    cfg = Config(seed=0, radius=5, frame_gap=4, reject_reflections=True, loss=loss,
                 noise_type="gaussian", noise_level=.4, lr=.003, batch_size=4, steps=4)
    model = build_model(cfg, torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    for _ in range(cfg.steps):
        batch = stream.sample(cfg.batch_size)
        optimizer.zero_grad(set_to_none=True)
        objective = reconstruction_loss(model(torch.from_numpy(batch.input)),
                                        torch.from_numpy(batch.target), cfg.loss)
        objective.backward()
        optimizer.step()
    with torch.inference_mode():
        held = MovieStream(np.random.SeedSequence([7, 0]), **cfg.movie_kwargs()).sample(16, need_target=False)
        z = model.encode(torch.from_numpy(held.input))
        output = model.decode(z)
    observed = {"model_sha256": _parameter_digest(model), "last_train_loss": objective.item(),
                "latent_sha256": tensor_sha(z), "output_sha256": tensor_sha(output)}
    assert observed == REFERENCE["short_training"][loss]


def test_variants_share_random_draws_and_change_only_prescribed_parameters():
    models, rng = {}, {}
    for variant in ("default", "a", "b"):
        models[variant] = k.build_model(k.config_for(5, 0., "l2", 0), variant, torch.device("cpu"))
        rng[variant] = torch.get_rng_state().clone()
    assert torch.equal(rng["default"], rng["a"]) and torch.equal(rng["a"], rng["b"])
    default, a, b = (models[v] for v in ("default", "a", "b"))
    for name, value in default.state_dict().items():
        if name not in ("decoder.9.weight", "decoder.9.bias"):
            assert torch.equal(value, a.state_dict()[name])
            assert torch.equal(value, b.state_dict()[name])
    assert torch.equal(default.decoder[-1].weight, a.decoder[-1].weight * .1)
    assert torch.equal(a.decoder[-1].weight, b.decoder[-1].weight)
    assert torch.count_nonzero(default.decoder[-1].bias).item() == 0
    assert torch.count_nonzero(a.decoder[-1].bias).item() == 0
    assert torch.count_nonzero(b.decoder[-1].bias).item() > 0
    x = torch.full((2, 2, 64, 64), .3)
    with torch.inference_mode():
        for variant, model in models.items():
            centered = x if variant == "b" else x - .3
            expected_z = model.to_latent(model.encoder(centered).flatten(1))
            assert torch.equal(model.encode(x), expected_z)
            raw = model.decoder(model.from_latent(expected_z).reshape(-1, 32, 4, 4))
            expected = raw if variant == "b" else .3 + raw
            assert torch.equal(model(x), expected)


def test_plan_and_historical_config_identities():
    runtime = k.runtime_info(torch.device("cpu"))
    jobs = k.plan()
    assert len(jobs) == 80 and len(set(cfg for _, cfg in jobs)) == 40
    specs = [k.endpoint_spec(cfg, variant, runtime) for variant, cfg in jobs]
    assert len({digest(spec) for spec in specs}) == 80
    assert {spec["runtime"]["initialization"] for spec in specs} == {"a", "b"}
    assert all("initialization" not in spec["config"] for spec in specs)
    assert all(cfg.steps == 3000 and cfg.batch_size == 128 and cfg.lr == .003 and
               cfg.frame_gap == 4 and cfg.reject_reflections and cfg.probe_test == 5000 and
               cfg.probe_train == 10000 and cfg.placement == "target_only" for _, cfg in jobs)
    for row in read_results().values():
        if row["purpose"] in ("experiment", "final_g_switch", "final_g_l2_anchor", "background_texture"):
            assert digest(identity(Config(**row["config"]), row["purpose"], row["runtime"])) == row["config_hash"]
    for previous, current, expected in ((.2, .3, "background_only"), (.65, .3, "past_only"),
                                       (.2, .65, "current_only"), (.65, .65, "both")):
        assert k.classification({"center_intensity_previous": previous,
                                 "center_intensity_current": current}) == expected


@pytest.mark.parametrize("variant", ("a", "b"))
def test_variant_checkpoint_resume_is_bitwise_identical(tmp_path, monkeypatch, variant):
    cfg = replace(k.config_for(5, .4, "l1", 2), steps=4, batch_size=2)
    spec = k.endpoint_spec(cfg, variant, k.runtime_info(torch.device("cpu")))
    monkeypatch.setattr(k, "CACHE", tmp_path / "resumed")
    k.train_to(cfg, spec, torch.device("cpu"), 2)
    _, resumed = k.train_to(cfg, spec, torch.device("cpu"), 4)
    monkeypatch.setattr(k, "CACHE", tmp_path / "direct")
    _, direct = k.train_to(cfg, spec, torch.device("cpu"), 4)
    assert resumed["model_sha256"] == direct["model_sha256"]
    assert resumed["last_train_loss"] == direct["last_train_loss"]
    assert resumed["stream_rng"] == direct["stream_rng"]
    assert resumed["accepted"] == direct["accepted"] == 8
    assert resumed["proposed"] == direct["proposed"]
    for key, value in direct["optimizer"]["state"].items():
        assert torch.equal(resumed["optimizer"]["state"][key]["exp_avg"], value["exp_avg"])
        assert torch.equal(resumed["optimizer"]["state"][key]["exp_avg_sq"], value["exp_avg_sq"])


def test_failed_condition_is_recorded_and_not_retrained(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(k, "FAILURES", tmp_path / "failures.jsonl")
    monkeypatch.setattr(k, "TRAJECTORY", tmp_path / "trajectory.jsonl")
    called = []
    def fail(*args):
        called.append(True)
        raise k.StudyFailure("synthetic nonfinite loss", 300)
    monkeypatch.setattr(k, "train_to", fail)
    cfg = k.config_for(5, 0., "l2", 0)
    runtime = k.runtime_info(torch.device("cpu"))
    assert k.run_one(cfg, "b", runtime, torch.device("cpu")) == "failed"
    assert k.run_one(cfg, "b", runtime, torch.device("cpu")) == "failed_previous"
    assert len(called) == 1
    row = next(iter(read_results(k.FAILURES).values()))
    assert row["observed_step"] == 300 and row["last_completed_step"] is None
    assert row["exception_type"] == "StudyFailure" and row["retry_performed"] is False
