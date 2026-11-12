import numpy as np
import pytest

from data import MovieStream
from diagnostic_probes import TemporalReconstruction, endpoint_scores, paired_scores, raw_centers


def test_rmse_is_measured_from_coordinates_and_r2_is_axis_mean():
    truth = np.array([[-1.0, -2.0], [1.0, 2.0]])
    scores = paired_scores(truth, truth + [1.0, 3.0])
    assert scores["rmse_x"] == 1 and scores["rmse_y"] == 3
    assert scores["rmse"] == pytest.approx(np.sqrt(5))
    assert scores["r2"] == (0 + (1 - 9/4)) / 2


def test_endpoint_error_cancellation_and_reflection_are_separate():
    actual = np.array([[[2.5, 3.5], [1.5, 3.0]], [[10., 10.], [12., 10.5]],
                       [[20., 20.], [21., 19.]]])
    velocity = np.array([[1., -.5], [2., .5], [1., -1.]])
    # A shared 2 px error on both endpoints cancels in their difference.
    scores = endpoint_scores(actual, actual + 2, velocity)
    assert scores["previous_position"]["rmse"] == scores["current_position"]["rmse"] == 2
    assert scores["delta_vs_displacement"]["rmse"] == 0
    assert scores["delta_vs_velocity"]["rmse"] == pytest.approx(np.sqrt(4/6))


@pytest.mark.parametrize("case,skill,ratio", [("same", 1.0, 1.0), ("static", 0.0, 0.0),
                                              ("backwards", -3.0, 1.0)])
def test_temporal_metric_uses_zero_change_reference(case, skill, ratio):
    batch = MovieStream(1723, radius=5).sample(4)
    predictions = {"same": batch.clean, "static": np.repeat(batch.clean[:, :1], 2, axis=1),
                   "backwards": batch.clean[:, ::-1]}
    metric = TemporalReconstruction()
    metric.update(batch.clean, predictions[case])
    assert metric.result()["r2_vs_zero_change"] == pytest.approx(skill)
    assert metric.result()["rms_ratio"] == pytest.approx(ratio)


def test_centroid_uses_only_observed_brightness_and_correct_axis_order():
    image = np.full((1, 2, 64, 64), .3, np.float32)
    image[0, 0, 21, 7] = 1.
    image[0, 1, 5, 8:10] = 1.
    np.testing.assert_allclose(raw_centers(image, .3), [[[7, 21], [8.5, 5]]])
