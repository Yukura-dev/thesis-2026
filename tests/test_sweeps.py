from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plot import aggregate
from run_sweep import (
    LOSSES, RADII, RATES, base_config, calibration_configs, calibration_table, canonical, digest,
    identity, read_results, run_one, select_rates, stage_configs,
)
from training import Config


def test_recommended_budget_preserves_historical_stage_zero(monkeypatch):
    args = SimpleNamespace(latent_dim=16, background=.3, foreground=1., steps=None)
    monkeypatch.setattr("run_sweep.SWEEP_STEPS", 1234)
    assert base_config(args).steps == 1234  # plotting and calibration
    args.stage = 1
    assert base_config(args).steps == 1234
    args.stage = 0
    assert base_config(args).steps == 300
    args.stage, args.steps = 3, 300
    assert base_config(args).steps == 300


def test_sweep_overlap_and_representatives():
    base, rates = Config(), dict.fromkeys(LOSSES, 1e-3)
    s0, s1, s2, s3 = [stage_configs(i, base, rates) for i in range(4)]
    assert list(map(len, (s0, s1, s2, s3))) == [1, 30, 60, 330]
    assert [len(set(s1) - set(s0)), len(set(s2) - set(s1)), len(set(s3) - set(s1 + s2))] == [29, 54, 246]
    assert [(c.noise_type, c.noise_level) for c in s2 if c.noise_level > 0] == sorted(
        [(c.noise_type, c.noise_level) for c in s2 if c.noise_level > 0],
        key=lambda cell: (cell[0] != "salt_pepper", cell[1]))
    reps = [c for c in s3 if c.placement != "target_only" or c.loss == "huber"]
    assert len(reps) == 30
    assert all(c.radius == 5 and c.noise_level == .2 for c in reps)
    assert canonical(replace(base, noise_type="salt_pepper", placement="both")) == canonical(base)
    assert all(c.seed not in (0, 1, 2) for c in calibration_configs(base))


def test_calibration_uses_independent_validation_per_loss():
    base, runtime = Config(), {"device": "cpu"}
    rows = {}
    for i, config in enumerate(calibration_configs(base)):
        score = .25 if config.lr == RATES[LOSSES.index(config.loss) % 3] else .5
        spec = identity(config, "calibration", runtime)
        rows[digest(spec)] = {**spec, "validation_loss": score}
    table = calibration_table(base, runtime, rows)
    assert select_rates(table) == {name: RATES[i % 3] for i, name in enumerate(LOSSES)}
    assert calibration_table(replace(base, latent_dim=4), runtime, rows) is None
    assert calibration_table(base, {"device": "mps"}, rows) is None


def test_torn_final_write_resume_skips_completed_run(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    calls = []
    def fake_run(c, device, calibration):
        calls.append((c, device, calibration))
        return {"metrics": {}, "timing": {"total_seconds": .1}}
    monkeypatch.setattr("training.run_experiment", fake_run)
    config, runtime = Config(), {"device": "cpu"}
    run_one(config, "experiment", runtime, 0, {}, path)
    saved = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"config": ')
    rows = read_results(path, repair=True)
    run_one(config, "experiment", runtime, 1, rows, path)
    assert len(calls) == 1
    assert path.read_bytes() == saved
    # An interrupted append without a trailing newline may contain a complete record.
    path.write_bytes(saved.rstrip(b"\n"))
    assert len(read_results(path, repair=True)) == 1
    assert path.read_bytes() == saved
    with path.open("ab") as stream:
        stream.write(b'{"bad":\n')
    with pytest.raises(ValueError, match="invalid complete record"):
        read_results(path, repair=True)


def test_difference_pairs_only_common_seeds_and_reports_sample_sd():
    cells = {}
    for loss, values in (("l1", {0: .9, 1: .2, 2: .5}), ("l2", {0: .4, 2: .4})):
        for seed, value in values.items():
            cells[5, "gaussian", 0., "target_only", loss, seed] = {"metric": value}
    means, sds, counts = aggregate(cells, "salt_pepper", "target_only", "metric")
    i = RADII.index(5)
    assert means[i, 0] == pytest.approx(.3)
    assert sds[i, 0] == pytest.approx(np.std([.5, .1], ddof=1))
    assert counts[i, 0] == 2
    assert np.isnan(means[i, 1]) and counts[i, 1] == 0
