import math

import pytest

from final_summary import (
    CENTERS, G_LEVELS, H_PROTOCOL, METRICS, METHODS, SEEDS,
    check_training_history, checked_metrics, digest, g_config, h_config,
    paired_comparison, statistics, switch_comparison,
)


def test_success_is_strict_and_retains_training_seed_ids():
    result = statistics([.8, .800000001, .7, .9, -.1], [10, 20, 30, 40, 50], probe=True)
    assert result["successful_seeds"] == [20, 40]
    assert result["unsuccessful_seeds"] == [10, 30, 50]
    assert result["successes"] == 2 and result["success_rate"] == .4
    assert result["median"] == .8
    assert result["mean"] == pytest.approx(.6200000002)
    assert result["sd"] == pytest.approx(math.sqrt(.66800000036 / 4))
    with pytest.raises(ValueError, match="unique seed"):
        statistics([.9, .7], [3, 3], probe=True)


def pair_records():
    records = []
    controls = [0., 1., .1, .9, .2]
    switched = [.3, 0., .4, .1, .5]
    for method, scores in zip(METHODS[:2], (controls, switched), strict=True):
        for seed, score in enumerate(scores):
            records.append(dict(radius=5, noise_level=.2, seed=seed, method=method,
                                config_hash=f"{method}-{seed}", anchor_config_hash=f"anchor-{seed}",
                                data_stream_state_sha256=f"stream-{seed}",
                                psnr=20 + 10 * score, corrected_psnr=21 + 10 * score,
                                velocity_r2_mlp=score))
    return records


def test_pair_seed_first_then_take_median_and_never_classify_difference_as_r2():
    # The median of paired differences (.3) differs from the difference of medians (.1).
    records = pair_records()
    seed_rows, cells = paired_comparison(list(reversed(records)))
    cell = next(c for c in cells if c["radius"] == 5 and c["noise_level"] == .2)
    scores = cell["metrics"]["velocity_r2_mlp"]
    assert [r["seed"] for r in seed_rows] == [0, 1, 2, 3, 4]
    assert scores["mean"] == pytest.approx(-.18)
    assert scores["median"] == pytest.approx(.3)
    assert scores["sd"] == pytest.approx(math.sqrt(.437))
    assert set(scores) == {"n", "mean", "sd", "median"}
    assert cell["complete"]
    assert [r["velocity_r2_mlp"] for r in seed_rows] == pytest.approx([.3, -1, .3, -.8, .3])
    records[-1]["anchor_config_hash"] = "different-anchor"
    with pytest.raises(ValueError, match="same three-thousand-step"):
        paired_comparison(records)


def test_pairing_rejects_different_continued_noise_streams():
    rows = pair_records()
    rows[-1]["data_stream_state_sha256"] = "changed-noise-stream"
    with pytest.raises(ValueError, match="complete data stream"):
        paired_comparison(rows)


def test_disappearance_requires_the_own_pre_switch_frame_to_have_been_visible():
    def record(seed, previous, current):
        values = dict.fromkeys(METRICS, .5)
        values.update(zip(CENTERS, (previous, current)))
        return dict(radius=5, noise_level=.2, seed=seed, method=METHODS[1],
                    config_hash=f"after-{seed}", model_sha256=f"weights-{seed}",
                    anchor_config_hash=f"before-{seed}", **values)

    records = [record(0, .649999999, .3), record(1, .65, .2)]
    anchors = {"before-0": {**record(0, .65, .64), "config_hash": "before-0"},
               "before-1": {**record(1, .9, .6), "config_hash": "before-1"}}
    comparisons, disappeared = switch_comparison(records, anchors)
    assert len(comparisons) == 2
    assert [(r["seed"], r["disappeared_frame"]) for r in disappeared] == [(0, "previous")]
    assert comparisons[0]["visible_before_previous"]
    assert not comparisons[0]["visible_before_current"]
    assert not comparisons[1]["disappeared_previous"]  # after == threshold remains visible
    assert not comparisons[1]["disappeared_current"]  # was already below threshold


def test_independent_design_matches_producer_identities():
    # Independently enumerated radius/corruption/seed/method cases; no producer plan used.
    from switch_study import anchor_spec, endpoint_spec, runtime_info
    from texture_study import experiment_spec
    from training import Config
    runtime = runtime_info("mps")
    keys = set()
    for radius, levels in G_LEVELS.items():
        for level in levels:
            for seed in SEEDS:
                cfg = Config(**g_config(radius, level, "l2_6000", seed))
                expected_anchor = dict(config=g_config(radius, level, "l2_6000", seed, anchor=True),
                                       purpose="final_g_l2_anchor", runtime=runtime)
                assert anchor_spec(cfg, runtime) == expected_anchor
                for method in METHODS:
                    expected = dict(config=g_config(radius, level, method, seed), purpose="final_g_switch",
                                    runtime={**runtime, "method": method})
                    assert endpoint_spec(cfg, method, runtime) == expected
                    keys.add(digest(expected))
    assert len(keys) == 120
    texture_runtime = {**runtime, "protocol": H_PROTOCOL, "texture_method": {"fixture": "no clipping"}}
    for radius in (5, 8):
        for texture in (.1, .4):
            for loss in ("l1", "l2"):
                for seed in SEEDS:
                    cfg = h_config(radius, texture, loss, seed)
                    assert experiment_spec(Config(**cfg), texture_runtime) == dict(
                        config=cfg, purpose="background_texture", runtime=texture_runtime)
    assert "background_texture_sd" not in h_config(5, 0., "l1", 0)


def test_reused_probe_metrics_and_optimizer_reset_metadata_are_checked():
    metrics = dict.fromkeys(METRICS, .7)
    source = dict(config_hash="source", model_sha256="same", metrics=metrics)
    row = dict(config=g_config(5, 0., METHODS[1], 3), config_hash="new", method=METHODS[1],
               model_sha256="same", metrics=dict(metrics), optimizer_reset_at=3000,
               training_phases=[[0, 3000, "l2"], [3000, 6000, "l1"]],
               training_seed_sequence=[0, 3], continued_motion_and_noise_streams=True,
               train_window_counts={"accepted": 768000},
               provenance=[dict(kind="measured_probes_reused_images_verified", source_config_hash="source",
                                model_sha256="same")])
    check_training_history(row, {"source": source})
    row["metrics"]["velocity_r2_mlp"] = .6
    with pytest.raises(ValueError, match="reused measurement"):
        check_training_history(row, {"source": source})
    row["metrics"] = dict(metrics)
    row["optimizer_reset_at"] = None
    with pytest.raises(ValueError, match="training history"):
        check_training_history(row, {"source": source})


def test_frame_pool_and_affine_scores_cannot_be_silently_inconsistent():
    values = dict.fromkeys(METRICS, .7)
    row = dict(config_hash="fixture", probe_splits=[6, 7], metrics=values)
    checked_metrics(row)
    values["center_intensity_previous"] = .2
    with pytest.raises(ValueError, match="Frame centers"):
        checked_metrics(row)
