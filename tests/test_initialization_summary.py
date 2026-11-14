import math

import pytest

from initialization_summary import (METRICS, PROTOCOL, PURPOSE, TRAJECTORY_PURPOSE,
                                    classify, collect, digest, expected_config, identify)


def endpoint():
    metrics = dict.fromkeys(METRICS, .7)
    metrics.update(psnr=20., corrected_psnr=21., background_mae=.01,
                   position_r2_mlp=.8, velocity_r2_mlp=.80000001,
                   future_r2_mlp=-.1)
    row = dict(config=expected_config(5, 0., "l1", 0),
                purpose=PURPOSE, runtime=dict(protocol=PROTOCOL, initialization="a",
                                              device="cpu", numpy="fixture", torch="fixture"),
                probe_splits=[6, 7], metrics=metrics, model_sha256="weights",
                frame_classification="both")
    row["config_hash"] = digest({k: row[k] for k in ("config", "purpose", "runtime")})
    return row


def point_for(source):
    return {**source, "purpose": TRAJECTORY_PURPOSE, "config_hash": "point", "step": 3000,
            "endpoint_config_hash": source["config_hash"],
            "runtime": {**source["runtime"], "checkpoint_step": 3000, "endpoint_config_hash": source["config_hash"]},
            "evaluation_seed_sequence": [7, 0], "evaluation_n": 5000,
            "metrics": {**source["metrics"], "output_mean": .3}}


def test_frame_classification_includes_exact_midpoint():
    assert classify(.64999999, .64999999) == "background_only"
    assert classify(.65, .64) == "past_only"
    assert classify(.64, .65) == "current_only"
    assert classify(.65, .65) == "both"
    with pytest.raises(ValueError):
        classify(math.nan, .7)


def test_controlled_lr_and_duplicate_environment_cannot_be_mixed():
    source = endpoint()
    assert len(identify({"a": source})) == 1
    source["config"]["lr"] = .001
    with pytest.raises(ValueError, match="controlled"):
        identify({"a": source})
    with pytest.raises(ValueError, match="Duplicate"):
        identify({"a": endpoint(), "b": endpoint()})


def test_missing_seeds_are_explicit_and_success_is_unrounded():
    summary = collect({"a": endpoint()}, {}, {}, {}, {}, allow_partial=True)
    assert not summary["complete"]
    assert len(summary["seeds"]) == 1 and len(summary["pending"]) == 119
    cell = next(c for c in summary["conditions"] if c["observed_seeds"])
    assert cell["counts"] == dict(background_only=0, past_only=0, current_only=0, both=1)
    assert cell["metrics"]["position_r2_mlp"]["successes"] == 0
    assert cell["metrics"]["velocity_r2_mlp"]["successes"] == 1
    assert cell["metrics"]["future_r2_mlp"]["mean"] == -.1
    with pytest.raises(ValueError, match="incomplete"):
        collect({"a": endpoint()}, {}, {}, {}, {})


def test_trajectory_end_must_be_the_same_weights_and_measurement():
    source = endpoint()
    point = {**point_for(source), "model_sha256": "different"}
    with pytest.raises(ValueError, match="weights disagree"):
        collect({"a": source}, {}, {}, {"b": point}, {}, allow_partial=True)
    point["model_sha256"] = source["model_sha256"]
    point["metrics"]["background_mae"] = .02
    with pytest.raises(ValueError, match="scores disagree"):
        collect({"a": source}, {}, {}, {"b": point}, {}, allow_partial=True)


def test_trajectory_requires_the_planned_condition_and_original_split():
    source = endpoint()
    point = point_for(source)
    point["config"] = {**point["config"], "radius": 3}
    with pytest.raises(ValueError, match="Unexpected"):
        collect({"a": source}, {}, {}, {"b": point}, {}, allow_partial=True)
    point = point_for(source)
    point["evaluation_seed_sequence"] = [9, 0]
    with pytest.raises(ValueError, match="evaluation split"):
        collect({"a": source}, {}, {}, {"b": point}, {}, allow_partial=True)
