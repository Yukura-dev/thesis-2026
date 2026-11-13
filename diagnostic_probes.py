"""Endpoint and velocity checks for the clean r=5 diagnostic, plus a raw-image control."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from data import MovieStream, SIZE, validate_frame_gap
from evaluation import ImageMetrics, TemporalReconstruction, grouped_r2, predict_probes, validate_probe_splits

ROOT = Path(__file__).resolve().parent


def paired_scores(actual, predicted):
    """Unweighted mean per-axis R², and actual per-axis / pooled coordinate RMSE."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mse = np.square(predicted - actual).mean(axis=0)
    var = actual.var(axis=0)
    if np.any(var == 0):
        raise ValueError("Cannot compute R² for a constant test target")
    r2 = 1 - mse / var
    return {"r2_x": float(r2[0]), "r2_y": float(r2[1]), "r2": float(r2.mean()),
            "rmse_x": float(np.sqrt(mse[0])), "rmse_y": float(np.sqrt(mse[1])),
            "rmse": float(np.sqrt(mse.mean()))}


def endpoint_scores(actual, predicted, velocities, frame_gap=1):
    validate_frame_gap(frame_gap)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1, 2, 2)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2, 2)
    delta = predicted[:, 1] - predicted[:, 0]
    return {"previous_position": paired_scores(actual[:, 0], predicted[:, 0]),
            "current_position": paired_scores(actual[:, 1], predicted[:, 1]),
            "delta_vs_displacement": paired_scores(actual[:, 1] - actual[:, 0], delta),
            "delta_vs_velocity": paired_scores(velocities, delta / frame_gap)}


def _batches(args, split, size, batch_size):
    stream = MovieStream(np.random.SeedSequence([split, 0]), **args)
    for start in range(0, size, batch_size):
        yield stream.sample(min(batch_size, size - start), need_target=False)


def evaluate_diagnostic(model, config, device, *, probe_splits=(2, 3)):
    """Fit historical six-output probes and separate four-output endpoint probes on z."""
    train_split, test_split = validate_probe_splits(probe_splits)
    metrics = ImageMetrics()
    temporal = TemporalReconstruction()

    def collect(split, size):
        z_values, targets, ends = [], [], []
        for batch in _batches(config.movie_kwargs(), split, size, config.batch_size):
            with torch.inference_mode():
                z = model.encode(torch.from_numpy(batch.input).to(device))
                z_values.append(z.cpu().numpy())
                if split == test_split:
                    reconstruction = model.decode(z).cpu().numpy()
                    metrics.update(reconstruction, batch, config.radius)
                    temporal.update(batch.clean, reconstruction)
            targets.append(batch.labels)
            ends.append(batch.centers.reshape(len(z), 4))
        return tuple(np.concatenate(items) for items in (z_values, targets, ends))

    model.eval()
    z, targets, ends = collect(train_split, config.probe_train)
    z_test, targets_test, ends_test = collect(test_split, config.probe_test)
    original = predict_probes(z, targets, z_test)
    endpoints = predict_probes(z, ends, z_test)
    direct = metrics.result()
    probe = {}
    for method, predictions in original.items():
        direct.update(grouped_r2(targets_test, predictions, method))
        by_target = {f"direct_{name}": paired_scores(targets_test[:, j:j+2], predictions[:, j:j+2])
                     for j, name in zip((0, 2, 4), ("position", "velocity", "future"))}
        by_target.update(endpoint_scores(ends_test, endpoints[method], targets_test[:, 2:4], config.frame_gap))
        probe[method] = by_target
    return {"direct": direct, "probe": probe, "temporal_reconstruction": temporal.result()}


def raw_image_features(images):
    """2x2 mean pool both full frames; every subsequent regression coefficient is learned."""
    return images.reshape(-1, 2, SIZE // 2, 2, SIZE // 2, 2).mean(axis=(3, 5)).reshape(len(images), -1)


def raw_centers(images, background):
    """Independent physical control using intensity-weighted pixel-center coordinates."""
    contrast = np.subtract(images, np.float32(background), dtype=np.float64)
    mass = contrast.sum(axis=(2, 3))
    pixel = np.arange(SIZE, dtype=np.float64)
    x = (contrast.sum(axis=2) @ pixel) / mass
    y = (contrast.sum(axis=3) @ pixel) / mass
    return np.stack((x, y), axis=-1)


def fit_raw_ridge(args, features, count, batch_size):
    """Fit image-only coefficients by ridge regression on the outgoing velocities."""
    start = 0
    x = None
    y = np.empty((count, 2), np.float64)
    for batch in _batches(args, 2, count, batch_size):
        block = features(batch.input)
        if x is None:
            x = np.empty((count, block.shape[1]), np.float64)
        stop = start + len(block)
        x[start:stop] = block
        y[start:stop] = batch.labels[:, 2:4]
        start = stop
    mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x -= mean
    std = np.sqrt(np.einsum("ij,ij->j", x, x) / count)
    std[std < 1e-12] = 1.0
    x /= std
    gram = x.T @ x
    rhs = x.T @ (y - y_mean)
    del x
    gram.flat[::gram.shape[0] + 1] += 1.0
    weights = np.linalg.solve(gram, rhs)
    return lambda images: (features(images) - mean) / std @ weights + y_mean


def raw_baseline(frame_gap=1):
    """Two supervised image-only readouts and an independent physical control.

    The primary ridge learns one free weight per pixel in each input frame.
    The 2x2 pooled ridge is equivalent to sharing weights within each 2x2
    block. Neither readout receives centers or displacement. The independent
    centroid calculation illustrates ambiguity when motion reflects at a wall.
    """
    validate_frame_gap(frame_gap)
    args = dict(radius=5, frame_gap=frame_gap, noise_type="gaussian", noise_level=0.0, noise_placement="target_only",
                background=0.3, foreground=1.0)
    n_train, n_test, batch_size = 10000, 5000, 128
    full = fit_raw_ridge(args, lambda imgs: imgs.reshape(len(imgs), -1), n_train, batch_size)
    pooled = fit_raw_ridge(args, raw_image_features, n_train, batch_size)
    predictions, pooled_predictions, centroid_estimates, ends, velocity = [], [], [], [], []
    for batch in _batches(args, 3, n_test, batch_size):
        predictions.append(full(batch.input))
        pooled_predictions.append(pooled(batch.input))
        centroid_estimates.append(raw_centers(batch.input, args["background"]))
        ends.append(batch.centers)
        velocity.append(batch.labels[:, 2:4])
    ends = np.concatenate(ends).astype(np.float64)
    velocities = np.concatenate(velocity)
    delta = ends[:, 1] - ends[:, 0]
    # Record reflection incidence and the effect of the true-position difference
    # itself as a predictor of outgoing velocity. Tolerance only absorbs float32
    # rounding when subtracting two positions (order 64) to obtain a velocity.
    reflected = (np.abs(delta / frame_gap - velocities) > 1e-5).any(axis=1)
    return {"protocol": "clean-two-frame-raw-control-v1", "config": args,
            "train_count": n_train, "test_count": n_test, "batch_size": batch_size,
            "train_seed_sequence": [2, 0], "test_seed_sequence": [3, 0],
            "feature": "all input pixels, both 64x64 frames flattened (8192 columns)",
            "pooled_feature": "2x2 spatial averages of both input frames flattened (2048 columns)",
            "target": "outgoing (vx, vy) at t", "regressor": "ridge, unpenalized intercept",
            "ridge_alpha": 1.0, "standardization": "training pixel feature mean and population SD",
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
            "velocity_from_raw_pixels": paired_scores(velocities, np.concatenate(predictions)),
            "velocity_from_2x2_pooled_pixels": paired_scores(velocities, np.concatenate(pooled_predictions)),
            "image_centroid_control": endpoint_scores(ends, np.concatenate(centroid_estimates), velocities, frame_gap),
            "true_delta_vs_velocity": paired_scores(velocities, delta / frame_gap),
            "test_examples_reflected_between_frames": int(reflected.sum()),
            "displacement_coordinate_rms": float(np.sqrt(np.square(delta).mean())),
            "velocity_coordinate_rms": float(np.sqrt(np.square(velocities, dtype=np.float64).mean()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the clean r=5 raw-image velocity reference")
    parser.add_argument("--raw-baseline", required=True, action="store_true")
    parser.add_argument("--frame-gap", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        name = ("raw_velocity_baseline.json" if args.frame_gap == 1
                else f"raw_velocity_baseline_gap{args.frame_gap}.json")
        args.output = ROOT / "results" / name
    result = raw_baseline(args.frame_gap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}: raw image velocity R² {result['velocity_from_raw_pixels']['r2']:.6f}")
