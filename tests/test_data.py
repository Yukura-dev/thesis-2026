import numpy as np
import pytest

from data import MovieStream, SIZE, disk_images, reflect


@pytest.mark.parametrize("radius", [1, 2, 3, 5, 8])
def test_disk_covers_correct_pixel_area_at_walls_and_between_pixels(radius):
    centers = np.array([[radius - 0.5, radius - 0.5], [31.137, 24.222],
                        [SIZE - 0.5 - radius, SIZE - 0.5 - radius]])
    image = disk_images(centers, radius, 0, 1)
    np.testing.assert_allclose(image.sum((1, 2)), np.pi * radius**2, rtol=2e-6)
    assert np.logical_and(image >= 0, image <= 1).all()


def test_subpixel_motion_changes_small_disk_continuously():
    xs = 31.11 + np.arange(4) * 0.002
    centers = np.column_stack((xs, np.full_like(xs, 24.32)))
    images = disk_images(centers, 1, 0, 1)
    changes = np.abs(np.diff(images, axis=0)).sum(axis=(1, 2))
    assert ((changes > 0) & (changes < .02)).all()
    weighted_x = (images * np.arange(SIZE)).sum(axis=(1, 2)) / images.sum(axis=(1, 2))
    assert np.all(np.diff(weighted_x) > 0)


def test_collision_and_future_labels():
    # r=2 means low=1.5, high=61.5. The two axes hit opposite walls.
    previous = np.array([60.5, 2.25])
    initial_v = np.array([3.0, -2.0])
    current, v = reflect(previous, initial_v, 1, 2)
    np.testing.assert_array_equal(current, [59.5, 2.75])
    np.testing.assert_array_equal(v, [-3.0, 2.0])
    future, _ = reflect(current, v, 5, 2)
    np.testing.assert_array_equal(future, [44.5, 12.75])
    # At an exact hit subsequent frames travel away from the wall.
    _, at_wall_v = reflect(np.array([60.5, 2.5]), np.array([1., -1.]), 1, 2)
    np.testing.assert_array_equal(at_wall_v, [-1., 1.])
    batch = MovieStream(13, radius=2).sample(128)
    np.testing.assert_array_equal(batch.labels[:, :2], batch.centers[:, 1])
    future, _ = reflect(batch.centers[:, 1], batch.labels[:, 2:4], 5, 2)
    np.testing.assert_allclose(future, batch.labels[:, 4:], atol=8e-6)


@pytest.mark.parametrize("kind,level", [("gaussian", .4), ("salt_pepper", .3)])
def test_noise_placement_reproducibility_and_independence(kind, level):
    def sample(where, need_target=True):
        return MovieStream(27, radius=5, noise_type=kind, noise_level=level,
                          noise_placement=where).sample(32, need_target=need_target)
    both, target, inp = (sample(k) for k in ("both", "target_only", "input_only"))
    np.testing.assert_array_equal(target.input, both.clean)
    np.testing.assert_array_equal(target.target, both.target)
    np.testing.assert_array_equal(inp.input, both.input)
    np.testing.assert_array_equal(inp.target, both.clean)
    np.testing.assert_array_equal(sample("both", False).input, both.input)
    assert not np.array_equal(both.target, both.input)
    first, second = both.input - both.clean, both.target - both.clean
    if kind == "gaussian":
        assert np.min(both.input) < 0 and np.max(both.input) > 1
        assert abs(first.mean()) < 0.005
        assert abs(first.std() - level) < 0.005
        assert abs(np.corrcoef(first.ravel(), second.ravel())[0, 1]) < .02
    else:
        bg = both.clean == .3
        for val in (0., 1.):
            assert abs((both.input[bg] == val).mean() - level/2) < .004


@pytest.mark.parametrize("frame_gap", [0, -1, 1.5, True])
def test_frame_gap_requires_positive_integer(frame_gap):
    from training import Config
    with pytest.raises(ValueError, match="positive integer"):
        MovieStream(0, radius=5, frame_gap=frame_gap)
    with pytest.raises(ValueError, match="positive integer"):
        Config(frame_gap=frame_gap)


def test_gap8_observes_the_same_trajectory_including_reflections():
    # Reconstruct the first eight one-frame transitions independently, without
    # using the closed-form reflect() called by MovieStream.
    sample = MovieStream(13, radius=5, frame_gap=8).sample(256)
    default = MovieStream(13, radius=5).sample(256)
    explicit_one = MovieStream(13, radius=5, frame_gap=1).sample(256)
    np.testing.assert_array_equal(default.clean, explicit_one.clean)
    np.testing.assert_array_equal(default.labels, explicit_one.labels)
    np.testing.assert_array_equal(sample.labels, default.labels)
    np.testing.assert_array_equal(sample.clean[:, 1], default.clean[:, 1])
    center = sample.centers[:, 1].astype(np.float64)
    velocity = -sample.labels[:, 2:4].astype(np.float64)
    lo, hi = 4.5, 58.5
    reflected = np.zeros(len(center), dtype=bool)
    for _ in range(8):
        center += velocity
        left, right = center <= lo, center >= hi
        reflected |= (left | right).any(axis=1)
        center[left] = 2*lo - center[left]
        center[right] = 2*hi - center[right]
        velocity[left | right] *= -1
    np.testing.assert_allclose(sample.centers[:, 0], center, atol=6e-6)
    assert reflected.any() and (~reflected).any()
    observed_speed = (sample.centers[:, 1] - sample.centers[:, 0]) / 8
    np.testing.assert_allclose(observed_speed[~reflected], sample.labels[~reflected, 2:4], atol=1e-6)
    assert np.max(np.abs(observed_speed[reflected] - sample.labels[reflected, 2:4])) > 1
