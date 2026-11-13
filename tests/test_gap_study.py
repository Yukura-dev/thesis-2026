"""Protocol decisions use all prescribed seeds, strict thresholds and ordered tie breaks."""
import pytest

from gap_study import (CALIBRATION, SELECTION, calibration_configs, selected_gap, selected_rates,
                       selection_configs, study_spec)
from run_sweep import digest


RUNTIME = {"device": "cpu", "protocol": "test"}


def rows_for(configs, purpose, scores):
    result = {}
    for cfg in configs:
        spec = study_spec(cfg, purpose, RUNTIME)
        result[digest(spec)] = {**spec, "metrics": scores(cfg)}
    return result


def test_gap_rule_considers_both_radii_strict_threshold_and_all_measurements():
    def scores(cfg):
        fails = (cfg.radius == 2 or
                 (cfg.frame_gap == 1 and cfg.radius == 8 and cfg.seed == 4) or
                 (cfg.frame_gap == 2 and cfg.radius == 5 and cfg.seed == 1))
        return {"velocity_r2_mlp": 0.8 if fails else 0.81}

    rows = rows_for(selection_configs(), SELECTION, scores)
    assert len(rows) == 60
    assert selected_gap(rows, RUNTIME) == 4
    for row in rows.values():
        row["metrics"]["velocity_r2_mlp"] = 0.8
    assert selected_gap(rows, RUNTIME) is None
    rows.popitem()
    with pytest.raises(ValueError, match="missing"):
        selected_gap(rows, RUNTIME)


def test_lr_rule_groups_three_seeds_filters_in_order_and_breaks_exact_tie():
    def scores(cfg):
        # Candidates' rank is computed from three seeds, so one high-scoring seed
        # cannot put the smallest LR back in contention after the velocity filter.
        velocity = {0.0003: (1.0, 0.2, 0.2), 0.001: (0.9,) * 3, 0.003: (0.894,) * 3}
        position = {0.0003: 1.0, 0.001: 0.92, 0.003: 0.94}
        if cfg.loss == "l1":
            position[0.001] = 0.94
        db = 70.0 if cfg.lr == 0.0003 else 30.0
        return {"velocity_r2_mlp": velocity[cfg.lr][cfg.seed - 1729],
                "position_r2_mlp": position[cfg.lr], "psnr": db}

    rows = rows_for(calibration_configs(4), CALIBRATION, scores)
    assert len(rows) == 36
    assert selected_rates(rows, RUNTIME, 4) == {
        "l1": 0.001, "l2": 0.003, "huber": 0.003, "charbonnier": 0.003}
