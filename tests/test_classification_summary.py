from collections import Counter

import pytest

from classification_summary import (
    CLASSES, LEVELS, PROBES, RADII, condition_counts, expanded_memberships,
    heatmap_data, modal_class, probe_distribution, unique_records,
)
from frame_classification import analysis_loss, classify_measurement


def record(key, loss, category, score, memberships, *, seed=0):
    return dict(model_identity=key, source_config_hash=key, analysis_loss=loss,
                classification=category, memberships=memberships, radius=5, seed=seed,
                **{name: score for name in PROBES})


def test_memberships_repeat_controls_but_never_weight_global_probe_distribution():
    rows = [record("flat", "l1", "background_only", .2,
                   [{"arm": "A", "role": "primary"}, {"arm": "H", "role": "flat_control"}]),
            record("noisy", "l1", "background_only", .8,
                   [{"arm": "B", "role": "primary"}])]
    assert len(expanded_memberships(rows)) == 3
    result = probe_distribution(rows, [])
    global_bg = next(r for r in result if r["analysis_loss"] == "all" and r["classification"] == "background_only")
    assert global_bg["n"] == 2
    assert global_bg["velocity_r2_mlp_median"] == .5  # repeating flat control would give .2
    assert global_bg["velocity_r2_mlp_min"] == .2
    assert global_bg["velocity_r2_mlp_max"] == .8
    with pytest.raises(ValueError, match="Repeated identity"):
        unique_records(rows + [rows[0]])


def test_switches_stay_separate_from_pure_l1_and_untrained_is_only_a_reference():
    assert analysis_loss({"method": "l2_3000_l1_3000", "config": {"loss": "l1"}}) == "l2_to_l1"
    rows = [record("l1", "l1", "background_only", .1, []),
            record("switch", "l2_to_l1", "background_only", .9, [])]
    baselines = [dict(radius=5, seed=seed, **{name: .7 for name in PROBES}) for seed in range(5)]
    result = probe_distribution(rows, baselines)
    pure = next(r for r in result if r["analysis_loss"] == "l1" and r["classification"] == "background_only")
    reference = next(r for r in result if r["population"] == "untrained_reference" and r["radius"] == 5)
    assert pure["n"] == 1 and pure["velocity_r2_mlp_median"] == .1
    assert reference["n"] == 5 and reference["classification"] == "not_applicable"
    assert sum(r["n"] for r in result if r["analysis_loss"] == "all") == 2


def test_textured_background_criterion_can_change_class_without_editing_raw_rule():
    metrics = dict(center_intensity_previous=.66, center_intensity_current=.64,
                   preoverlay_background_center_previous=.4, preoverlay_background_center_current=.2)
    value = classify_measurement(metrics, foreground=1., textured=True)
    assert value["classification"] == "current_only"
    assert value["raw_comparison_classification"] == "past_only"
    assert value["signed_margin_previous"] == pytest.approx(-.04)
    assert value["signed_margin_current"] == pytest.approx(.04)
    assert value["borderline_previous"] and value["borderline_current"]
    at_boundary = dict(center_intensity_previous=.75, center_intensity_current=.625,
                       preoverlay_background_center_previous=.5, preoverlay_background_center_current=.25)
    assert classify_measurement(at_boundary, foreground=1., textured=True)["classification"] == "both"


def test_heatmap_reports_every_tied_mode_instead_of_arbitrary_first():
    counts = dict(zip(CLASSES, (1, 2, 2, 0), strict=True))
    assert modal_class(counts) == ["past_only", "current_only"]
    cells = [dict(arm="A", analysis_loss=loss, radius=radius, noise_level=level,
                  complete=True, n=5, **counts)
             for loss in ("l1", "l2") for radius in RADII for level in LEVELS]
    result = heatmap_data(cells)
    assert len(result) == 50
    assert all(r["tied"] and r["modes"] == ["past_only", "current_only"] for r in result)
    assert all(r["annotation"] == "Past / Now\n2/5 each" for r in result)
    assert modal_class(dict.fromkeys(CLASSES, 0)) == []


def test_condition_counts_preserve_control_role_and_missing_seeds():
    config = dict(radius=5, seed=0, noise_type="gaussian", noise_level=.4,
                  placement="target_only", loss="l1", steps=3000)
    memberships = [{"arm": "A", "role": "primary"}, {"arm": "D", "role": "target_only_control"}]
    targets = [dict(source=dict(config={**config, "seed": seed}), memberships=memberships) for seed in range(5)]
    rows = [dict(config, seed=seed, analysis_loss="l1", final_loss="l1", method="standard",
                 background_texture_sd=0., memberships=memberships, classification="background_only") for seed in range(4)]
    result = condition_counts(rows, targets)
    assert len(result) == 2
    assert Counter(r["role"] for r in result) == {"primary": 1, "target_only_control": 1}
    assert all(r["observed_seeds"] == [0, 1, 2, 3] and r["expected_seeds"] == list(range(5)) for r in result)
    assert all(not r["complete"] and r["background_only"] == 4 for r in result)
