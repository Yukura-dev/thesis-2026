import numpy as np
import pytest

from collapse_replay import AffineMoments


def test_global_affine_matches_one_pooled_least_squares_across_frames_and_batches():
    rng = np.random.default_rng(42)
    x = (0.3 + rng.normal(0, .0001, size=(7, 2, 8, 8))).astype(np.float32)
    y = .8 - 1.3 * x.astype(np.float64) + rng.normal(0, .00015, size=x.shape)
    # The common slope/intercept must use all samples, including the shorter tail.
    acc = AffineMoments()
    for i in range(0, len(x), 3):
        acc.update(x[i:i+3], y[i:i+3])
    actual = acc.coefficients()
    design = np.stack((x.ravel().astype(np.float64), np.ones(x.size)), axis=1)
    expected = np.linalg.lstsq(design, y.ravel(), rcond=None)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9)
    assert np.sqrt(acc.sxx/acc.n) == pytest.approx(x.std(dtype=np.float64), rel=1e-12)
    assert np.mean((actual[0]*x.astype(np.float64)+actual[1]-y)**2) <= np.mean((x-y)**2)


def test_constant_float32_reconstruction_uses_mean_clean_brightness():
    x = np.full(20, .3, dtype=np.float32)
    y = np.linspace(0, 1, len(x))
    acc = AffineMoments()
    acc.update(x[:7], y[:7])
    acc.update(x[7:], y[7:])
    a, b = acc.coefficients()
    assert a == 0 and b == pytest.approx(np.mean(y))
    assert acc.sxx == 0
