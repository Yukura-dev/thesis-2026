"""G design identity and exact optimizer/data-stream recovery without long fits."""
from dataclasses import replace

import numpy as np
import torch

import switch_study as g
from run_sweep import digest
from training import Config, _parameter_digest


def test_g_plan_is_exact_and_hashes_are_distinct():
    runtime = g.runtime_info(torch.device("cpu"))
    assert len(g.plan()) == 120
    assert len(g.conditions()) == 40
    assert {(r, p) for r, p, _ in g.conditions()} == {
        (5, 0.), (5, .1), (5, .2), (5, .3), (5, .45), (8, 0.), (8, .2), (8, .45)}
    specs = [g.endpoint_spec(g.config_for(r,p,s), m, runtime) for r,p,s,m in g.plan()]
    assert len({digest(s) for s in specs}) == 120
    for spec in specs:
        cfg = Config(**spec["config"])
        assert (cfg.steps,cfg.lr,cfg.frame_gap,cfg.reject_reflections,cfg.batch_size) == (6000,.003,4,True,128)
        assert cfg.placement == "target_only"
        assert cfg.seed in range(5)


def test_training_resume_preserves_optimizer_noise_and_motion(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CACHE", tmp_path)
    device = torch.device("cpu")
    cfg = replace(g.config_for(5,.2,3), steps=2, batch_size=2)
    runtime = g.runtime_info(device)
    spec = g.identity(cfg,"test",runtime)
    _, one, _ = g.train_to(cfg,spec,device,1)
    _, resumed, start = g.train_to(cfg,spec,device,2)
    assert start == 1
    spec2 = {**spec,"purpose":"independent-test"}
    _, direct, _ = g.train_to(cfg,spec2,device,2)
    assert resumed["model_sha256"] == direct["model_sha256"]
    assert resumed["last_train_loss"] == direct["last_train_loss"]
    assert resumed["stream_rng"] == direct["stream_rng"]
    assert resumed["accepted"] == direct["accepted"]
    assert resumed["proposed"] == direct["proposed"]
    assert torch.equal(resumed["torch_rng"], direct["torch_rng"])
    assert all(torch.equal(resumed["optimizer"]["state"][p]["exp_avg"],
                           direct["optimizer"]["state"][p]["exp_avg"])
               for p in direct["optimizer"]["state"])

    # Switching changes only Adam history, preserving weights and every RNG.
    before = {k: v["exp_avg"].clone() for k,v in one["optimizer"]["state"].items()}
    same_model, same_optimizer, same_stream = g.restore_training(cfg,device,one)
    for value in same_optimizer.state.values():
        value["exp_avg"].add_(123.)
    for key, old in before.items():
        assert torch.equal(one["optimizer"]["state"][key]["exp_avg"], old)
    warm_model, warm_optimizer, warm_stream = g.restore_training(cfg,device,one,reset_optimizer=True)
    assert _parameter_digest(same_model) == _parameter_digest(warm_model) == one["model_sha256"]
    assert len(same_optimizer.state) > 0 and len(warm_optimizer.state) == 0
    old_batch, warm_batch = same_stream.sample(2), warm_stream.sample(2)
    np.testing.assert_array_equal(old_batch.input,warm_batch.input)
    np.testing.assert_array_equal(old_batch.target,warm_batch.target)
    np.testing.assert_array_equal(old_batch.labels,warm_batch.labels)
