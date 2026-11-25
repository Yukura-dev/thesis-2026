from dataclasses import replace
import math

import pytest
import torch

from run_sweep import config_record, digest, read_results
from smoothing_study import (BETA_PER_SIGMA, classify, experiment_spec, export, historical_reference,
                             plan, run_one, runtime_info)
from training import Config, gaussian_smoothed_l1, reconstruction_loss


@pytest.mark.parametrize("sigma", [.02, .1, .4])
def test_gauss_l1_is_expected_l1_under_gaussian_target_noise(sigma):
    gen = torch.Generator().manual_seed(0)
    error = torch.linspace(-1, 1, 41, dtype=torch.float64)
    noise = sigma * torch.randn(400_000, 1, generator=gen, dtype=torch.float64)
    monte_carlo = (error + noise).abs().mean(0)
    torch.testing.assert_close(gaussian_smoothed_l1(error, sigma), monte_carlo, atol=4e-3 * sigma + 1e-4, rtol=0)


def test_gauss_l1_gradient_is_erf_and_curvature_matches_smooth_l1():
    sigma = .1
    error = torch.linspace(-.5, .5, 101, dtype=torch.float64, requires_grad=True)
    gaussian_smoothed_l1(error, sigma).sum().backward()
    torch.testing.assert_close(error.grad, torch.special.erf(error.detach() / (sigma * math.sqrt(2))))
    # Curvature at zero: sqrt(2/pi)/sigma for gauss_l1, 1/beta for unit-slope Huber.
    h = 1e-4
    zero = torch.tensor([-h, 0., h], dtype=torch.float64)
    for name, scale in (("gauss_l1", sigma), ("smooth_l1", sigma * BETA_PER_SIGMA)):
        values = torch.stack([reconstruction_loss(e.view(1), torch.zeros(1, dtype=torch.float64), name, scale)
                              for e in zero])
        curvature = (values[0] - 2 * values[1] + values[2]) / h**2
        assert curvature.item() == pytest.approx(math.sqrt(2 / math.pi) / sigma, rel=1e-3)
    # Far from zero both have unit slope like L1.
    large = torch.tensor([.9], dtype=torch.float64, requires_grad=True)
    reconstruction_loss(large, torch.zeros(1, dtype=torch.float64), "smooth_l1", .2).backward()
    assert large.grad.item() == pytest.approx(1.)


def test_scaled_losses_validate_scale_and_old_hashes_are_unchanged():
    with pytest.raises(ValueError):
        Config(loss="gauss_l1")
    with pytest.raises(ValueError):
        Config(loss="l1", loss_scale=.1)
    with pytest.raises(ValueError):
        reconstruction_loss(torch.zeros(1), torch.zeros(1), "smooth_l1")
    assert "loss_scale" not in config_record(Config())
    assert config_record(Config(loss="gauss_l1", loss_scale=.1))["loss_scale"] == .1
    # Every completed historical row keeps its recorded hash under the new Config.
    rows = read_results()
    checked = 0
    for key, row in rows.items():
        if row["purpose"] in ("experiment", "background_texture") and \
                row["runtime"].get("protocol") == "circle-ae-filtered-final-v1":
            cfg = Config(**row["config"])
            assert config_record(cfg) == row["config"]
            checked += 1
    assert checked > 100


def test_plan_pairs_three_arms_per_sigma_and_reuses_no_rows():
    cases = plan()
    assert len(cases) == 10 + 10 + 5 * 2 * 3 * 5 == 170
    assert len({digest(experiment_spec(c[-1], {"device": "cpu"})) for c in cases}) == 170
    by = {(a, r, s, seed): cfg for a, r, s, seed, cfg in cases}
    noisy, smooth, gauss = by[("noisy_l1", 5, .2, 3)], by[("smooth_l1", 5, .2, 3)], by[("gauss_l1", 5, .2, 3)]
    assert (noisy.loss, noisy.noise_level, noisy.placement) == ("l1", .2, "target_only")
    assert gauss.noise_level == 0 and gauss.loss_scale == .2
    assert smooth.noise_level == 0 and smooth.loss_scale == pytest.approx(.2 * math.sqrt(math.pi / 2))
    assert replace(gauss, loss="l1", loss_scale=0., noise_level=.2) == noisy


def test_historical_reference_finds_all_mps_controls_with_existing_sigmas():
    found = historical_reference(read_results())
    for r in (5, 8):
        for sigma in (0., .05, .1, .2, .4):
            assert all(("noisy_l1", r, sigma, s) in found for s in range(5))
        assert all(("l2", r, 0., s) in found for s in range(5))
    assert ("noisy_l1", 5, .02, 0) not in found
    assert all(row["config"]["lr"] == .003 and row["config"]["frame_gap"] == 4 for row in found.values())


def test_classification_uses_both_frames():
    assert classify({"center_intensity_previous": .65, "center_intensity_current": .3}) == "past"
    assert classify({"center_intensity_previous": .3, "center_intensity_current": .9}) == "current"
    assert classify({"center_intensity_previous": .3, "center_intensity_current": .3}) == "background"


def test_short_cpu_run_records_and_resumes(tmp_path):
    runtime = runtime_info(torch.device("cpu"))
    cfg = replace(plan()[-1][-1], steps=2, probe_train=64, probe_test=32)
    ledger, lock = tmp_path / "runs.jsonl", tmp_path / "gpu.lock"
    row = run_one(cfg, runtime, ledger, lock, store=False)
    assert row["config"]["loss"] == "smooth_l1" and row["purpose"] == "final_m_loss_smoothing"
    assert {"center_intensity_previous", "corrected_psnr"} <= set(row["metrics"])
    assert run_one(cfg, runtime, ledger, lock, store=False) == row
    assert len(ledger.read_text().splitlines()) == 1
    assert export(read_results(ledger), runtime, tmp_path / "out") == 0  # shortened config is not in plan
