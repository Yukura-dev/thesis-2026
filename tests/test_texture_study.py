from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from data import MovieStream, disk_images, smooth_backgrounds
from evaluation import evaluate
from run_sweep import config_record, digest, historical_config, read_results
from texture_study import configs, experiment_spec, run_one, runtime_info
from training import Config, _validate
from untrained_encoder import experiment_spec as untrained_spec


@pytest.mark.parametrize("sd", [.1, .4])
def test_texture_has_requested_mean_sd_and_smoothness_before_overlay(sd):
    textures = smooth_backgrounds(np.random.default_rng(7), 12, .3, sd)
    np.testing.assert_allclose(textures.mean((1, 2), dtype=np.float64), .3, atol=2e-9)
    np.testing.assert_allclose(textures.std((1, 2), dtype=np.float64), sd, atol=3e-9)
    # Adjacent values are close after the requested blur; samples are independent.
    assert np.square(np.diff(textures, axis=2)).mean() < .1 * sd**2
    assert abs(np.corrcoef(textures[0].ravel(), textures[1].ravel())[0, 1]) < .3
    if sd == .4:
        assert textures.min() < 0 and textures.max() > 1


def test_texture_is_shared_by_frames_and_supervision_and_disk_is_opaque():
    opts = dict(radius=5, frame_gap=4, reject_reflections=True, background_texture_sd=.4)
    stream = MovieStream(8, **opts)
    batch = stream.sample(8)
    np.testing.assert_array_equal(batch.input, batch.clean)
    np.testing.assert_array_equal(batch.target, batch.clean)
    # Rebuild geometry at original float64 precision, independently of rendering.
    geometry = MovieStream(8, **opts).geometry(8)[0]
    coverage = disk_images(geometry, 5, 0, 1)
    background_both = (coverage[:, 0] == 0) & (coverage[:, 1] == 0)
    np.testing.assert_array_equal(batch.clean[:, 0][background_both], batch.clean[:, 1][background_both])
    np.testing.assert_array_equal(batch.clean[coverage == 1], np.ones((coverage == 1).sum(), np.float32))
    # Fractional boundary pixels use alpha compositing, not an additive bright disk.
    texture = smooth_backgrounds(MovieStream(8, **opts).texture_rng, 8, .3, .4)[:, None]
    expected = texture * (1 - coverage) + coverage
    np.testing.assert_array_equal(batch.clean, expected)
    assert ((coverage > 0) & (coverage < 1)).any()


def test_texture_rng_preserves_motion_and_noise_streams_and_chunking():
    opts = dict(radius=8, frame_gap=4, reject_reflections=True, noise_type="gaussian",
                noise_level=.2, noise_placement="both")
    plain = MovieStream(9, **opts)
    textured = MovieStream(9, **opts, background_texture_sd=.4)
    a, b = plain.sample(32), textured.sample(32)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.centers, b.centers)
    for name in ("motion", "input_rng", "target_rng"):
        assert getattr(plain, name).bit_generator.state == getattr(textured, name).bit_generator.state
    whole = MovieStream(12, radius=5, background_texture_sd=.1).sample(21)
    stream = MovieStream(12, radius=5, background_texture_sd=.1)
    chunks = [stream.sample(n) for n in (7, 14)]
    np.testing.assert_array_equal(whole.clean, np.concatenate([b.clean for b in chunks]))


@pytest.mark.parametrize("sd", [-.1, float("inf"), float("nan")])
def test_bad_texture_sd_is_rejected(sd):
    with pytest.raises(ValueError, match="background_texture_sd"):
        Config(background_texture_sd=sd)
    with pytest.raises(ValueError, match="background_texture_sd"):
        MovieStream(0, radius=5, background_texture_sd=sd)


def test_zero_texture_keeps_old_serialized_identities():
    cfg = Config(reject_reflections=True, frame_gap=4)
    assert "background_texture_sd" not in config_record(cfg)
    assert "background_texture_sd" not in historical_config(cfg)
    assert "background_texture_sd" not in cfg.movie_kwargs()
    assert "background_texture_sd" not in untrained_spec(cfg, {})["config"]
    nonzero = replace(cfg, background_texture_sd=.1)
    assert config_record(nonzero)["background_texture_sd"] == .1
    # Every archived Config-based config retains its exact shape on round trip,
    # apart from untrained rows whose steps=0/loss omission are their own schema.
    for row in read_results().values():
        c = row["config"]
        if "loss" in c and "background_texture_sd" not in c:
            normalized = config_record(Config(**c))
            if "frame_gap" not in c:
                normalized.pop("frame_gap", None)
            assert normalized == c


def test_validation_and_probe_metrics_use_textured_clean_reference(monkeypatch):
    class OffsetIdentity(torch.nn.Module):
        def encode(self, x):
            return x
        def decode(self, z):
            return z + .025
        def forward(self, x):
            return self.decode(self.encode(x))
    model = OffsetIdentity()
    cfg = Config(background_texture_sd=.4, reject_reflections=True, frame_gap=4)
    assert abs(_validate(model, cfg, torch.device("cpu")) - .025**2) < 1e-9
    # Isolate the image reference path: training neural probes is covered elsewhere.
    monkeypatch.setattr("evaluation.fit_probes", lambda *args: {})
    metrics = evaluate(model, **cfg.movie_kwargs(), n_train=8, n_test=8, batch_size=4)
    assert abs(metrics["psnr"] - (-10*np.log10(.025**2))) < 1e-5


def test_h_design_and_resume_skips_recorded_fit(tmp_path, monkeypatch):
    cases = configs()
    assert len(cases) == len(set(cases)) == 40
    assert {c.radius for c in cases} == {5, 8}
    assert {c.background_texture_sd for c in cases} == {.1, .4}
    assert all(c.steps == 3000 and c.lr == .003 and c.frame_gap == 4 and
               c.reject_reflections and c.noise_level == 0 and c.latent_dim == 16 for c in cases)
    calls = []
    def fake_fit(cfg, device, **kwargs):
        calls.append((cfg, kwargs))
        return {"metrics": {"psnr": 20., "velocity_r2_mlp": .5},
                "timing": {"total_seconds": 1.}}
    monkeypatch.setattr("texture_study.run_experiment", fake_fit)
    path, lock = tmp_path/"runs.jsonl", tmp_path/"gpu.lock"
    cfg, runtime = cases[0], runtime_info("cpu")
    first = run_one(cfg, runtime, path, lock)
    second = run_one(cfg, runtime, path, lock)
    assert first == second and len(calls) == 1
    assert calls[0][1]["probe_splits"] == (6, 7) and calls[0][1]["affine"] is True
    assert first["config_hash"] == digest(experiment_spec(cfg, runtime))
