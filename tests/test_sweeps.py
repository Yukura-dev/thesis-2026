from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plot import aggregate, main_cases, summary_cells
from run_sweep import (
    FILTERED_PROTOCOL, LOSSES, MAIN_PROTOCOL, PROTOCOL, RADII, RATES, base_config, calibration_configs, calibration_table, canonical, digest,
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
    assert list(map(len, (s0, s1, s2, s3))) == [1, 50, 100, 330]
    assert [len(set(s1) - set(s0)), len(set(s2) - set(s1)), len(set(s3) - set(s1 + s2))] == [49, 90, 246]
    assert len(main_cases(base, rates)) == 386
    sensitivity = main_cases(base, rates, shared_lr=True)
    assert len(sensitivity) == 100 and {c.lr for c in sensitivity} == {1e-3}
    assert set(c.loss for c in sensitivity) == set(LOSSES)
    assert len(set(sensitivity) - set(s1)) == 50
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
    def fake_run(c, device, calibration, **kwargs):
        calls.append((c, device, calibration))
        return {"metrics": {}, "timing": {"total_seconds": .1}}
    monkeypatch.setattr("training.run_experiment", fake_run)
    config, runtime = Config(), {"device": "cpu", "protocol": PROTOCOL}
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


def test_archived_gap_one_hash_and_new_gap_identity():
    old_runtime = {"protocol": PROTOCOL, "device": "mps", "torch": "2.12.1", "numpy": "2.3.5"}
    cfg = Config(radius=5, loss="l2", steps=300, lr=.003)
    old = identity(cfg, "experiment", old_runtime)
    assert "frame_gap" not in old["config"]
    assert digest(old) == "01f3df4a5f4332aa093bbdaf4d05ce0658ee3453caa5975afa9b7fc7e987cb7c"
    new = identity(cfg, "experiment", dict(old_runtime, protocol=MAIN_PROTOCOL))
    assert new["config"]["frame_gap"] == 1 and digest(new) != digest(old)
    other = identity(replace(cfg, frame_gap=2), "experiment", new["runtime"])
    assert digest(other) != digest(new)


def test_median_and_strict_success_fraction_on_incomplete_five_seed_cell():
    rates = dict.fromkeys(LOSSES, .001)
    cases = main_cases(Config(), rates)
    key = (5, "gaussian", 0., "target_only", "l2")
    metrics = ("position_r2_ridge", "velocity_r2_ridge", "position_r2_mlp", "velocity_r2_mlp")
    data = {(*key, seed): dict.fromkeys(metrics, score) for seed, score in ((0, .81), (3, .80), (4, -.1))}
    rows = summary_cells(cases, data)
    partial = next(c for c in rows if c["n_evaluated"] > 0)
    assert partial["intended_seeds"] == [0, 1, 2, 3, 4] and not partial["complete"]
    assert partial["summary"]["velocity_r2_mlp"] == {"median": .8, "successes": 1, "success_rate": 1/3}
    assert all(c["summary"]["position_r2_mlp"]["success_rate"] is None
               for c in rows if not c["n_evaluated"])
    result, _, n = aggregate(data, "gaussian", "target_only", "position_r2_mlp", "l2", "success")
    assert result[RADII.index(5), 0] == 1/3 and n[RADII.index(5), 0] == 3


def test_filtered_protocol_isolated_and_sanity_strict():
    from run_sweep import SANITY_PURPOSE, filtered_sanity_config, require_passed_sanity
    from gap_study import study_spec, SELECTION
    rates = dict.fromkeys(LOSSES, .003)
    cfg = filtered_sanity_config(rates, 2)
    rt = {"device": "mps", "protocol": FILTERED_PROTOCOL}
    key = digest(identity(cfg, SANITY_PURPOSE, rt))
    legacy = replace(cfg, reject_reflections=False)
    assert "reject_reflections" not in study_spec(legacy, SELECTION, rt)["config"]
    assert identity(cfg, SANITY_PURPOSE, rt)["config"]["reject_reflections"] is True
    assert key != digest(identity(legacy, SANITY_PURPOSE, rt))
    assert len(main_cases(cfg, rates)) == 140
    stage3 = stage_configs(3, cfg, rates)
    assert len(stage3) == 330 and {c.seed for c in stage3} == {0, 1, 2}
    assert all(c.reject_reflections for c in stage3)
    with pytest.raises(ValueError, match="has not run"):
        require_passed_sanity({}, cfg, rt)
    row = {"probe_splits": [8, 9], "passed_velocity_sanity": True, "metrics": {"velocity_r2_mlp": .8}}
    with pytest.raises(ValueError, match="failed"):
        require_passed_sanity({key:row}, cfg, rt)
    row["metrics"]["velocity_r2_mlp"] = .800001
    assert require_passed_sanity({key:row}, cfg, rt) == row
    row["probe_splits"] = [6, 7]
    with pytest.raises(ValueError, match="wrong holdout"):
        require_passed_sanity({key:row}, cfg, rt)


def test_main_cli_defaults_filter_history_and_explicit_design_overrides(monkeypatch):
    import sys
    import run_sweep as command
    seen, gates = [], []
    monkeypatch.setattr(command, "read_results", lambda: {})
    monkeypatch.setattr(command, "runtime_info", lambda device, historical=False, filtered=False:
                        {"protocol": command.FILTERED_PROTOCOL if filtered else command.PROTOCOL if historical
                         else command.MAIN_PROTOCOL})
    monkeypatch.setattr(command, "chosen_gap", lambda rows, rt, override=None: override or 2)
    monkeypatch.setattr(command, "main_rates", lambda base, rt, rows: dict.fromkeys(LOSSES, .001))
    monkeypatch.setattr(command, "require_passed_sanity", lambda rows, cfg, rt: gates.append((cfg, rt)))
    original = command.stage_configs
    def record(stage, base, rates, **kw):
        seen.append((stage, base))
        return original(stage, base, rates, **kw)
    monkeypatch.setattr(command, "stage_configs", record)
    for flags in (["--stage", "0"], ["--stage", "1", "--latent-dim", "4", "--background", ".2",
                                      "--foreground", ".9", "--steps", "1234"],
                  ["--include-observed-reflections", "--stage", "2"], ["--reject-reflections", "--stage", "3"]):
        monkeypatch.setattr(sys, "argv", ["run_sweep.py", "--dry-run", *flags])
        command.main()
    assert [cfg.reject_reflections for _, cfg in seen] == [False, True, False, True]
    assert [(cfg.latent_dim, cfg.background, cfg.foreground, cfg.steps) for _, cfg in seen][1] == (4,.2,.9,1234)
    assert len(gates) == 2
    assert all(cfg.latent_dim == 16 and cfg.steps == command.SWEEP_STEPS for cfg, _ in gates)


def test_plain_plot_shows_filtered_screen_with_no_qualified_gap(tmp_path, monkeypatch, capsys):
    import sys
    import plot
    calls = []
    monkeypatch.setattr(plot, "read_results", lambda path: {})
    def draw(rows, rt, out):
        calls.append((rt["protocol"], out))
        return True
    monkeypatch.setattr(plot, "draw_selection_if_complete", draw)
    def none_qualified(*_):
        raise ValueError("No frame gap passed")
    monkeypatch.setattr(plot, "chosen_gap", none_qualified)
    for flags in ([], ["--include-observed-reflections"]):
        monkeypatch.setattr(sys, "argv", ["plot.py", "--out", str(tmp_path), *flags])
        plot.main()
    assert calls == [(FILTERED_PROTOCOL, tmp_path), (MAIN_PROTOCOL, tmp_path)]
    assert not list(tmp_path.iterdir())
    assert capsys.readouterr().out.count("No main-stage heatmaps: No frame gap passed") == 2


def test_corrected_psnr_joins_only_verified_source_weights_and_counts_partial_seeds():
    from collapse_replay import spec_for
    from plot import with_reconstruction_metrics
    cfg = Config(reject_reflections=True, frame_gap=4, steps=3000, lr=.003)
    runtime = {"device":"cpu", "protocol": FILTERED_PROTOCOL}
    spec = identity(cfg, "experiment", runtime)
    probes = dict.fromkeys(("position_r2_ridge", "velocity_r2_ridge", "position_r2_mlp", "velocity_r2_mlp"), .5)
    source = {**spec, "config_hash": digest(spec), "model_sha256": "example", "probe_splits": [6,7],
              "metrics": {**probes, "psnr":20.}}
    diagnostic = spec_for(source)
    extra = {"corrected_psnr":23., "affine_a":2., "affine_b":-.2, "background_psnr":21.,
             "output_pixel_sd":.02, "mean_frame_pixel_sd":.018}
    obs = {**diagnostic, "source_config_hash":source["config_hash"], "model_sha256":"example",
           "probe_score_stream":[7,0], "original_validation_exact":True,
           "original_reconstruction_metrics_exact":True, "metrics":{"psnr":20., **extra}}
    archive = {digest(diagnostic):obs}
    combined = with_reconstruction_metrics(source, archive)
    assert combined["corrected_psnr"] == 23. and "corrected_psnr" not in source["metrics"]
    other = {**source, "config_hash":"other-fit"}
    assert with_reconstruction_metrics(other, archive) == source["metrics"]
    metric_key = (5, "gaussian", 0., "target_only", "l2")
    data = {(*metric_key, 0):combined, (*metric_key, 1):source["metrics"]}
    summary = summary_cells([cfg, replace(cfg, seed=1)], data)[0]["summary"]
    assert summary["corrected_psnr"] == {"mean":23., "sd":None, "n":1}
    assert summary["psnr"] == {"mean":20., "sd":0., "n":2}
    for invalid in ({"model_sha256":"wrong-weights"}, {"probe_score_stream":[3,0]}):
        with pytest.raises(ValueError, match="does not match"):
            with_reconstruction_metrics(source, {digest(diagnostic):{**obs, **invalid}})
