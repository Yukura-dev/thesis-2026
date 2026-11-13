"""Frozen-encoder evaluation on independent, fixed motion and noise streams."""
import math

import numpy as np
import torch
from torch import nn

from data import MovieStream, SIZE


def grouped_r2(truth, prediction, probe):
    """Uniform mean of the two individual coordinate R² scores in each task."""
    truth = np.asarray(truth, dtype=np.float64)
    residual = np.square(truth - prediction).sum(axis=0)
    variance = np.square(truth - truth.mean(axis=0)).sum(axis=0)
    if np.any(variance == 0):
        raise ValueError("R² requires evaluation targets with nonzero variance")
    scores = 1 - residual / variance
    return {f"{task}_r2_{probe}": float(scores[i:i + 2].mean())
            for i, task in zip((0, 2, 4), ("position", "velocity", "future"))}


def _fit_mlp(x, y, xt):
    """Fit the same fixed two-hidden-layer probe for all conditions."""
    features = torch.from_numpy(x.astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32))
    test_features = torch.from_numpy(xt.astype(np.float32))
    # On these small matrices one CPU thread also makes reproducibility inexpensive.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(107)
            shuffle = torch.Generator().manual_seed(108)
            model = nn.Sequential(nn.Linear(features.shape[1], 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, targets.shape[1]))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(80):
                for ids in torch.randperm(len(x), generator=shuffle).split(256):
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(features[ids]), targets[ids])
                    loss.backward()
                    optimizer.step()
            with torch.inference_mode():
                return model(test_features).numpy()
    finally:
        torch.set_num_threads(threads)


def predict_probes(z_train, y_train, z_test):
    """Covariates and targets both use moments from the probe training set only."""
    def standardize(train, test=None):
        train = np.asarray(train, dtype=np.float64)
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale < 1e-12] = 1.0
        if test is None:
            return (train - mean) / scale, mean, scale
        return (train - mean) / scale, (test - mean) / scale

    x, xt = standardize(z_train, z_test)
    y, mean, scale = standardize(y_train)
    # Ridge alpha=1, unpenalized intercept restored via training means.
    weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y)
    return {"ridge": (xt @ weights) * scale + mean,
            "mlp": _fit_mlp(x, y, xt) * scale + mean}


def fit_probes(z_train, y_train, z_test, y_test):
    predictions = predict_probes(z_train, y_train, z_test)
    return {**grouped_r2(y_test, predictions["ridge"], "ridge"),
            **grouped_r2(y_test, predictions["mlp"], "mlp")}


class ImageMetrics:
    """Pool clean errors over pixels; average reconstruction contrasts over frames."""
    def __init__(self, *, include_center_intensity=False):
        self.include_center_intensity = include_center_intensity
        self.center_sum = 0.0
        self.squared_error = self.object_error = self.background_error = 0.0
        self.object_pixels = self.background_pixels = self.frames = 0
        self.contrast = 0.0

    def update(self, reconstruction, batch, radius):
        truth = batch.clean
        signed_error = reconstruction - truth
        error = np.abs(signed_error)
        self.squared_error += np.square(signed_error, dtype=np.float64).sum()
        # A pixel belongs to the disk if any positive area of its square overlaps it.
        pixels = np.arange(SIZE)
        centers = batch.centers.reshape(-1, 2)
        edge_x = np.maximum(np.abs(pixels - centers[:, 0, None]) - 0.5, 0)
        edge_y = np.maximum(np.abs(pixels - centers[:, 1, None]) - 0.5, 0)
        disk = edge_x[:, None, :]**2 + edge_y[:, :, None]**2 < radius**2
        images = reconstruction.reshape(-1, SIZE, SIZE)
        error = error.reshape(images.shape)
        self.object_error += error.sum(where=disk, dtype=np.float64)
        self.background_error += error.sum(where=~disk, dtype=np.float64)
        disk_pixels = disk.sum(axis=(1, 2))
        self.object_pixels += int(disk_pixels.sum())
        self.background_pixels += int(images.size - disk_pixels.sum())
        background = images.sum(axis=(1, 2), where=~disk, dtype=np.float64) / (SIZE**2 - disk_pixels)
        # Bilinear interpolation at both true subpixel centers.
        lower = np.floor(centers).astype(np.int64)
        dx, dy = (centers - lower).T
        x, y = lower.T
        k = np.arange(len(images))
        center = ((1 - dy) * ((1 - dx) * images[k, y, x] + dx * images[k, y, x + 1]) +
                  dy * ((1 - dx) * images[k, y + 1, x] + dx * images[k, y + 1, x + 1]))
        if self.include_center_intensity:
            self.center_sum += float(center.sum())
        self.contrast += float((center - background).sum())
        self.frames += len(images)

    def result(self):
        mse = self.squared_error / (self.object_pixels + self.background_pixels)
        result = {"psnr": -10 * math.log10(mse) if mse > 0 else math.inf,
                "object_mae": float(self.object_error / self.object_pixels),
                "background_mae": float(self.background_error / self.background_pixels),
                "center_contrast": self.contrast / self.frames}
        if self.include_center_intensity:
            result["center_intensity"] = self.center_sum / self.frames
        return result


class TemporalReconstruction:
    """Test-pixel temporal change, with zero predicted change as the baseline."""
    def __init__(self):
        self.squared_truth = self.squared_prediction = self.squared_error = 0.0

    def update(self, truth, prediction):
        actual = np.subtract(truth[:, 1], truth[:, 0], dtype=np.float64)
        predicted = np.subtract(prediction[:, 1], prediction[:, 0], dtype=np.float64)
        self.squared_truth += np.square(actual).sum()
        self.squared_prediction += np.square(predicted).sum()
        self.squared_error += np.square(predicted - actual).sum()

    def result(self):
        if self.squared_truth == 0:
            raise ValueError("Temporal change metrics need at least one changing pixel")
        return {"r2_vs_zero_change": float(1 - self.squared_error / self.squared_truth),
                "rms_ratio": float(np.sqrt(self.squared_prediction / self.squared_truth))}


def validate_probe_splits(splits):
    if (len(splits) != 2 or splits[0] == splits[1] or
            any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2 for k in splits)):
        raise ValueError("Probe splits must be two distinct integer namespaces >= 2 (0,1 are reserved)")
    return tuple(splits)


def evaluate(model, *, radius, noise_type="gaussian", noise_level=0.0,
             noise_placement="target_only", background=0.3, foreground=1.0,
             device="cpu", batch_size=128, n_train=10000, n_test=5000, frame_gap=1,
             temporal=False, probe_splits=(2, 3), reject_reflections=False):
    """Run both probes and reconstruction metrics once. No test data select weights."""
    train_split, test_split = validate_probe_splits(probe_splits)
    data_args = dict(radius=radius, noise_type=noise_type, noise_level=noise_level, frame_gap=frame_gap,
                     reject_reflections=reject_reflections,
                     noise_placement=noise_placement, background=background, foreground=foreground)
    metrics = ImageMetrics(include_center_intensity=reject_reflections)
    changes = TemporalReconstruction() if temporal else None

    def collect(split, size, decode):
        stream = MovieStream(np.random.SeedSequence([split, 0]), **data_args)
        z_values, labels = [], []
        with torch.inference_mode():
            for start in range(0, size, batch_size):
                batch = stream.sample(min(batch_size, size - start), need_target=False)
                x = torch.from_numpy(batch.input).to(device)
                z = model.encode(x)
                z_values.append(z.cpu().numpy())
                labels.append(batch.labels)
                if decode:
                    reconstruction = model.decode(z).cpu().numpy()
                    metrics.update(reconstruction, batch, radius)
                    if changes is not None:
                        changes.update(batch.clean, reconstruction)
        return np.concatenate(z_values), np.concatenate(labels)

    model.eval()
    z_train, y_train = collect(train_split, n_train, False)
    z_test, y_test = collect(test_split, n_test, True)
    result = {**metrics.result(), **fit_probes(z_train, y_train, z_test, y_test)}
    if changes is not None:
        result["temporal_reconstruction"] = changes.result()
    return result
