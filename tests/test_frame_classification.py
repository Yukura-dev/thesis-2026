from copy import deepcopy

import numpy as np
import pytest
import torch

from data import MovieStream
from frame_classification import (
    CLASSES, audit, borderline_frames, classify_frames, classify_measurement,
    design, load_saved, sample_centers, sample_with_background, saved_checkpoint,
)
from run_sweep import config_record, read_results
from training import Autoencoder, Config, _parameter_digest


def test_classes_are_frame_specific_and_threshold_is_inclusive():
    assert classify_frames(.3, .3) == "background_only"
    assert classify_frames(.65, .3) == "past_only"
    assert classify_frames(.3, .65) == "current_only"
    assert classify_frames(.65, .65) == "both"
    assert classify_frames(np.nextafter(.65, -np.inf), .65) == "current_only"
    assert borderline_frames(.65, .701) == (True, False)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            classify_frames(bad, .65)


def test_textured_rule_retains_raw_comparison_and_uses_local_midpoint():
    def metrics(prediction, background):
        return {f"{kind}_{frame}": value for frame in ("previous", "current")
                for kind, value in (("center_intensity", prediction),
                                    ("preoverlay_background_center", background))}
    case = classify_measurement(metrics(.65, .4), textured=True)
    assert case["classification"] == "background_only"
    assert case["raw_comparison_classification"] == "both"
    assert case["signed_margin_previous"] == pytest.approx(-.05)
    opposite = classify_measurement(metrics(.63, .2), textured=True)
    assert opposite["classification"] == "both"
    assert opposite["raw_comparison_classification"] == "background_only"
    flat = classify_measurement(metrics(.65, np.float32(.3)), textured=False)
    assert flat["classification"] == "both" and flat["signed_margin_previous"] == 0.


def test_bilinear_sampling_recovers_affine_images():
    yy, xx = np.mgrid[:64, :64]
    images = np.broadcast_to(xx+2*yy, (3, 2, 64, 64))
    centers = np.array([[[12.1, 23.4], [15.7, 28.3]]]*3, dtype=np.float32)
    sampled = sample_centers(images, centers)
    np.testing.assert_allclose(sampled, centers[:, :, 0]+2*centers[:, :, 1], atol=1e-5)


def test_preoverlay_texture_replay_never_advances_movie_or_noise_rngs():
    options = dict(radius=8, frame_gap=4, reject_reflections=True, background_texture_sd=.4,
                   noise_type="gaussian", noise_level=.2, noise_placement="both")
    stream, control = MovieStream([7,0], **options), MovieStream([7,0], **options)
    for count in (19, 7):
        before = deepcopy(stream.texture_rng.bit_generator.state)
        batch, background = sample_with_background(stream, count)
        original = control.sample(count, need_target=False)
        for name in ("input", "clean", "labels", "centers"):
            np.testing.assert_array_equal(getattr(batch, name), getattr(original, name))
        np.testing.assert_array_equal(background[:, 0], background[:, 1])
        np.testing.assert_allclose(background[:, 0].mean((1,2), dtype=np.float64), .3, atol=3e-9)
        np.testing.assert_allclose(background[:, 0].std((1,2), dtype=np.float64), .4, atol=3e-9)
        assert stream.texture_rng.bit_generator.state != before
        for name in ("motion", "texture_rng", "input_rng", "target_rng"):
            assert getattr(stream, name).bit_generator.state == getattr(control, name).bit_generator.state


def test_saved_only_loader_rejects_missing_or_wrong_weights(tmp_path, monkeypatch):
    source = {"config_hash": "absent", "purpose": "experiment"}
    monkeypatch.setattr("frame_classification.ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="未実行"):
        saved_checkpoint(source)
    cfg = Config(frame_gap=4, reject_reflections=True, steps=3000)
    model = Autoencoder(cfg.latent_dim, cfg.background)
    source.update(config=config_record(cfg), model_sha256="incorrect")
    path = tmp_path/"checkpoints"/"stage3"/"absent"/"weights.pt"
    path.parent.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "source_config_hash": "absent"}, path)
    with pytest.raises(ArithmeticError, match="SHA"):
        load_saved(source, torch.device("cpu"))
    source["model_sha256"] = _parameter_digest(model)
    loaded, _, actual_path = load_saved(source, torch.device("cpu"))
    assert actual_path == path and _parameter_digest(loaded) == source["model_sha256"]
    assert all(not p.requires_grad for p in loaded.parameters())


def test_design_has_514_unique_models_and_preserves_arm_seed_counts():
    targets = design(read_results(), "mps")
    assert len(targets) == len({t["model_identity"] for t in targets}) == 514
    assert not any(t["source"]["purpose"] == "final_g_l2_anchor" for t in targets)
    expected = {"A": 250, "B": 60, "D": 60, "G": 120, "H": 60}
    assert audit(targets)["memberships"] == expected
    assert audit(targets)["saved_checkpoints"] == 514
    b = [t for t in targets if any(m["arm"] == "B" for m in t["memberships"])]
    assert {t["source"]["config"]["seed"] for t in b} == {0, 1, 2}
    assert len([t for t in targets if {"arm":"H", "role":"flat_control"} in t["memberships"]]) == 20
    assert len([t for t in targets if {"arm":"D", "role":"target_only_control"} in t["memberships"]]) == 20
