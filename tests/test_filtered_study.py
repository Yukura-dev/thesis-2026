import pytest

import filtered_study as study
from gap_study import study_spec
from run_sweep import digest


def test_filtered_selection_requires_complete_new_protocol_and_both_radii():
    run = study.runtime_info("cpu")
    configs = study.selection_configs()
    rows = {}
    assert len(configs) == 30
    for cfg in configs:
        spec = study_spec(cfg, study.SELECTION, run)
        assert spec["config"]["reject_reflections"] is True
        below = cfg.radius == 2 or (cfg.frame_gap == 2 and cfg.radius == 8 and cfg.seed == 3)
        rows[digest(spec)] = {**spec, "metrics": {"velocity_r2_mlp": .8 if below else .81}}
    assert study.selected_gap(rows, run) == 4
    rows.popitem()
    with pytest.raises(ValueError, match="missing"):
        study.selected_gap(rows, run)


def test_oracle_gate_checks_numeric_results_and_expected_folds(tmp_path):
    import json
    path = tmp_path / "geometry.json"
    data = {"protocol": "observed-no-reflection-v1", "reject_reflections": True,
            "fit": {"n": 10000, "seed_sequence": [2, 0]}, "test": {"n": 5000, "seed_sequence": [3, 0]},
            "conditions": [{"frame_gap": g, "radius": r, "oracle": {"mlp": {"r2": .95}}}
                           for g in (2, 4) for r in (2, 5, 8)]}
    path.write_text(json.dumps(data))
    assert study.oracle_passed(path)
    data["conditions"][-1]["oracle"]["mlp"]["r2"] = .949999
    path.write_text(json.dumps(data))
    assert not study.oracle_passed(path)
    data["test"]["seed_sequence"] = [5, 0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol"):
        study.oracle_passed(path)
