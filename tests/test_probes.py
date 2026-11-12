import numpy as np
import pytest

from data import MovieStream
from evaluation import ImageMetrics, fit_probes


@pytest.fixture(scope="module")
def targets():
    def get(split, count):
        # Same stream, split and counts as the scientific probe evaluation.
        stream = MovieStream(np.random.SeedSequence([split, 0]), radius=5)
        return np.concatenate([stream.sample(min(128, count - start),
                                             need_target=False).labels
                               for start in range(0, count, 128)])
    return get(2, 10000), get(3, 5000)


def test_true_labels_give_unit_r2(targets):
    train, test = targets
    scores = fit_probes(train, train, test, test)
    print("true-label probe R²", scores)
    assert all(score > .99 for score in scores.values())


def test_shuffled_labels_give_chance_r2(targets):
    train, test = targets
    permutation = np.random.default_rng(931).permutation(len(train))
    scores = fit_probes(train, train[permutation], test, test)
    print("shuffled-label probe R²", scores)
    assert all(score < .02 for score in scores.values())


def test_image_metrics_use_true_support_both_frames_and_interpolated_center():
    batch = MovieStream(14, radius=5).sample(4)
    metrics = ImageMetrics()
    metrics.update(batch.clean, batch, 5)
    result = metrics.result()
    assert np.isposinf(result["psnr"])
    assert result["object_mae"] == result["background_mae"] == 0
    assert result["center_contrast"] == pytest.approx(.7)
    metrics = ImageMetrics()
    metrics.update(np.full_like(batch.clean, .3), batch, 5)
    result = metrics.result()
    assert result["background_mae"] == result["center_contrast"] == 0
    assert result["object_mae"] > 0
    reference = -10 * np.log10(np.square(batch.clean - .3, dtype=np.float64).mean())
    assert result["psnr"] == pytest.approx(reference)
