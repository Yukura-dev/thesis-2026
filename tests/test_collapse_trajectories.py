import numpy as np

from collapse_trajectories import ANCHOR, LONG, SMALL, WARM, plan
from data import MovieStream
from evaluation import ImageMetrics


def test_plan_visits_all_budgets_and_warm_sources_in_order():
    jobs = plan()
    assert len(jobs) == len(set(jobs)) == 60
    for seed in range(5):
        for radius in (5, 8):
            assert [step for r, s, family, step in jobs if (r, s, family) == (radius, seed, LONG)] == [
                1000, 3000, 10000, 30000]
        assert [step for r, s, family, step in jobs if (r, s, family) == (2, seed, SMALL)] == [3000, 10000]
        assert jobs.index((5, seed, ANCHOR, 3000)) < jobs.index((5, seed, WARM, 6000))


def test_separate_frame_centers_keep_original_metrics_and_channel_order():
    batch = MovieStream(13, radius=5, frame_gap=4, reject_reflections=True).sample(8)
    # Previous reconstruction places the disk correctly. Current reconstruction contains only background.
    prediction = batch.clean.copy()
    prediction[:, 1] = .3
    base = ImageMetrics(include_center_intensity=True)
    split = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
    for stats in (base, split):
        stats.update(prediction, batch, 5)
    old = base.result()
    result = split.result()
    assert np.isclose(result.pop("center_intensity_previous"), 1.)
    assert np.isclose(result.pop("center_intensity_current"), .3)
    assert result == old
