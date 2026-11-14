import pytest

from run_sweep import FILTERED_PROTOCOL, digest, identity, stage3_arms
from stage3_summary import METRICS, collect, mixed_seeds, statistics
from training import Config


def test_summary_separates_low_contrast_and_reuses_only_the_matching_target_controls():
    cfg = Config(frame_gap=4, steps=3000, reject_reflections=True)
    rates = {"l1": .003, "l2": .003}
    rt = {"protocol": FILTERED_PROTOCOL, "device": "cpu"}
    arms = stage3_arms(cfg, rates)
    rows = {}
    for arm, configs in arms.items():
        for c in configs:
            spec = identity(c, "experiment", rt)
            key = digest(spec)
            metrics = dict.fromkeys(METRICS, .8)
            metrics.update(center_intensity=c.foreground, velocity_r2_mlp=.800001 if c.seed == 0 else .8)
            rows[key] = {**spec, "config_hash": key, "model_sha256":"weights", "metrics":metrics,
                         "probe_splits":[6,7], "first_stage":3}
    from dataclasses import replace
    for c in arms["D"]:
        c = replace(c, placement="target_only")
        spec = identity(c,"experiment",rt)
        key = digest(spec)
        if key not in rows:  # salt-pepper r=5 seeds 3,4 for both losses are only D reference memberships
            rows[key] = {**spec,"config_hash":key,"model_sha256":"reference","probe_splits":[6,7],
                         "metrics":dict.fromkeys(METRICS,.9),"first_stage":2}
    data = collect(rows, {}, rt, cfg, rates)
    assert data["expected_unique_experiments"] == data["observed_unique_experiments"] == 404
    assert len(data["seeds"]) == 420 and len(data["conditions"]) == 92
    assert sum(1 for k in data["conditions"] if k["arm"] == "D" and k["placement"] == "target_only") == 4
    for group in data["conditions"]:
        assert group["complete"]
        assert group["metrics"]["center_intensity"]["mean"] == pytest.approx(
            .65 if group["arm"] == "C" else .96 if group["arm"] == "D" and group["noise_type"] == "salt_pepper"
            and group["placement"] == "target_only" else 1.)
        assert len(group["observed_seeds"]) == (3 if group["arm"] == "B" else 5)
        assert group["untrained"]["velocity_r2_mlp"] is None
    arm_b_mixed = [m for m in mixed_seeds(data) if m["arm"] == "B"]
    assert len(arm_b_mixed) == 20
    for group in arm_b_mixed:
        assert group["metric"] == "velocity_r2_mlp"
        assert group["successful_seeds"] == [0] and group["unsuccessful_seeds"] == [1,2]
    first = next(iter(rows))
    del rows[first]
    partial = collect(rows, {}, rt, cfg, rates)
    assert not partial["conditions"][0]["complete"]
    assert partial["observed_unique_experiments"] == 403


def test_statistics_uses_strict_threshold_and_seed_mean_median_sample_sd():
    res = statistics([1., .8, -.5], [4, 2, 0], True)
    assert res["mean"] == pytest.approx(1.3/3) and res["median"] == .8
    assert res["successes"] == 1 and res["successful_seeds"] == [4]
    assert res["unsuccessful_seeds"] == [2,0]
    assert res["success_rate"] == 1/3
    assert statistics([],[],True) is None
