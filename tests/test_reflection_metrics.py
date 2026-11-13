import numpy as np
import pytest

from reflection_metrics import bayes_endpoints, reflection_axes, stratified_velocity


def test_hit_includes_current_wall_but_excludes_previous_wall():
    # Radius 5 -> bounds [4.5,58.5]. Final state with outgoing +2 at left
    # wall or −3 at right wall corresponds to an included impact at t.
    pos = np.array([[4.5, 58.5], [8.5, 52.5], [7.5, 53.5]])
    vel = np.array([[2., -3.], [2., -3.], [2., -3.]])
    np.testing.assert_array_equal(reflection_axes(pos, vel, 5, 2),
                                  [[True, True], [False, False], [True, True]])


def test_bayes_averages_feasible_direct_and_reflected_histories_per_axis():
    # Gap 4, L=12: x at 6.5->8.5 comes from v=+0.5 or +1.5; y at
    # 56.5->54.5 from v=-0.5 or -1.5. Interior direct speeds are unique.
    ends = np.array([[[6.5, 56.5], [8.5, 54.5]], [[20., 22.], [28., 18.]]])
    predicted, n = bayes_endpoints(ends, 5, 4)
    np.testing.assert_array_equal(predicted, [[1., -1.], [2., -1.]])
    np.testing.assert_array_equal(n, [[2, 2], [1, 1]])


def test_reflecting_group_is_any_axis_and_centers_its_own_r2_denominator():
    truth = np.array([[0., 0.], [1., 2.], [4., 8.], [5., 10.]])
    predicted = truth + 1
    hits = np.array([[True, False], [False, True], [False, False], [False, False]])
    groups = stratified_velocity(truth, predicted, hits)
    assert groups["reflecting"]["n"] == groups["nonreflecting"]["n"] == 2
    assert groups["reflecting"]["r2"] == pytest.approx((-3. + 0.) / 2)
    assert groups["all"]["r2"] > 0
