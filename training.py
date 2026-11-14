"""One deterministic autoencoder fit and its held-out evaluation."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

from data import MovieStream, validate_frame_gap
from evaluation import evaluate, validate_probe_splits


@dataclass(frozen=True)
class Config:
    radius: int = 5
    noise_type: str = "gaussian"
    noise_level: float = 0.0
    placement: str = "target_only"
    loss: str = "l2"
    seed: int = 0
    latent_dim: int = 16
    steps: int = 300
    batch_size: int = 128
    lr: float = 1e-3
    background: float = 0.3
    foreground: float = 1.0
    probe_train: int = 10000
    probe_test: int = 5000
    frame_gap: int = 1
    reject_reflections: bool = False
    background_texture_sd: float = 0.0

    def __post_init__(self):
        validate_frame_gap(self.frame_gap)
        if not isinstance(self.reject_reflections, bool):
            raise ValueError("reject_reflections must be boolean")
        if self.radius not in (1, 2, 3, 5, 8) or self.latent_dim not in (4, 16, 64):
            raise ValueError("Unsupported radius or latent dimension")
        if self.loss not in ("l1", "l2", "huber", "charbonnier"):
            raise ValueError(f"Unknown reconstruction loss: {self.loss}")
        if self.noise_type not in ("gaussian", "salt_pepper"):
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        if self.placement not in ("input_only", "target_only", "both"):
            raise ValueError(f"Unknown noise placement: {self.placement}")
        if not np.isfinite(self.noise_level) or self.noise_level < 0:
            raise ValueError("Noise level must be finite and nonnegative")
        if self.noise_type == "salt_pepper" and self.noise_level > 1:
            raise ValueError("Salt-and-pepper probability must not exceed 1")
        if self.steps < 1 or self.batch_size < 1 or self.seed < 0:
            raise ValueError("Use positive steps and batch size, and a nonnegative seed")
        if min(self.probe_train, self.probe_test) < 2 or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Probe sets need at least two samples. LR must be positive and finite")
        if not (0 <= self.background <= 1 and 0 <= self.foreground <= 1):
            raise ValueError("Clean image intensities must be in [0, 1]")
        if self.foreground == self.background:
            raise ValueError("Equal foreground and background make the object invisible")
        if not np.isfinite(self.background_texture_sd) or self.background_texture_sd < 0:
            raise ValueError("background_texture_sd must be finite and nonnegative")

    def movie_kwargs(self):
        options = dict(radius=self.radius, noise_type=self.noise_type, noise_level=self.noise_level,
                       noise_placement=self.placement, background=self.background, foreground=self.foreground,
                       frame_gap=self.frame_gap, reject_reflections=self.reject_reflections)
        if self.background_texture_sd:
            options["background_texture_sd"] = self.background_texture_sd
        return options


class Autoencoder(nn.Module):
    """4x stride-2 convolution to a 4x4 grid, dense bottleneck, mirrored decoder."""

    def __init__(self, latent_dim: int, background: float):
        super().__init__()
        self.background = background
        widths = (2, 8, 16, 24, 32)
        enc = []
        for a, b in zip(widths, widths[1:]):
            enc.extend([nn.Conv2d(a, b, 4, 2, 1, bias=False), nn.GroupNorm(1, b),
                        nn.LeakyReLU(0.1)])
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Linear(32 * 4 * 4, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 32 * 4 * 4)
        dec = []
        for a, b in zip(reversed(widths[1:]), reversed(widths[:-1])):
            dec.append(nn.ConvTranspose2d(a, b, 4, 2, 1, bias=b == 2))
            if b != 2:
                dec.extend([nn.GroupNorm(1, b), nn.LeakyReLU(0.1)])
        self.decoder = nn.Sequential(*dec)
        # Start every loss near the same background, with an open gradient path to z.
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.1)
            self.decoder[-1].bias.zero_()

    def encode(self, x):
        return self.to_latent(self.encoder(x - self.background).flatten(1))

    def decode(self, z):
        grid = self.from_latent(z).reshape(-1, 32, 4, 4)
        return self.background + self.decoder(grid)

    def forward(self, x):
        return self.decode(self.encode(x))


def reconstruction_loss(prediction, target, name):
    if name == "l2":
        return F.mse_loss(prediction, target)
    if name == "l1":
        return F.l1_loss(prediction, target)
    if name == "huber":
        return F.huber_loss(prediction, target, delta=0.1)
    if name == "charbonnier":
        # Standard Charbonnier penalty with epsilon 0.001 in [0, 1] brightness units.
        return ((prediction - target).square() + 1e-6).sqrt().mean()
    raise ValueError(f"Unknown loss: {name}")


def resolve_device(device="auto"):
    if str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _configure(seed):
    # Called before creating the first CUDA tensor or module.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class _PeakMemory:
    """Sample this fit's RSS and MPS driver allocations; CUDA has an allocator counter."""

    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.rss = 0
        self.mps = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _sample(self):
        self.rss = max(self.rss, self.process.memory_info().rss)
        if self.device.type == "mps":
            self.mps = max(self.mps, torch.mps.driver_allocated_memory())

    def _watch(self):
        while not self.stop.wait(0.05):
            self._sample()

    def __enter__(self):
        self._sample()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()

    def result(self):
        values = {"peak_rss_mib": self.rss / 2**20, "sample_interval_seconds": 0.05}
        if self.device.type == "mps":
            values["peak_mps_driver_mib"] = self.mps / 2**20
        if self.device.type == "cuda":
            values["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        return values


@torch.no_grad()
def _validate(model, cfg, device):
    # Only this independent split selects the LR, using the candidate's own criterion.
    # Pilots are clean. In a noisy sweep this value also refers to the clean reference.
    model.eval()
    stream = MovieStream(np.random.SeedSequence([1, 0]), **cfg.movie_kwargs())
    n = 1024
    total = 0.0
    for start in range(0, n, cfg.batch_size):
        count = min(cfg.batch_size, n - start)
        batch = stream.sample(count, need_target=False)
        x = torch.from_numpy(batch.input).to(device)
        clean = torch.from_numpy(batch.clean).to(device)
        total += reconstruction_loss(model(x), clean, cfg.loss).item() * count
    return total / n


def _parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_experiment(config: Config, device="auto", *, calibration=False, diagnostic=False,
                   temporal=False, probe_splits=(2, 3), affine=False, store_weights=None):
    """Train once. Calibration omits test probes and returns the selection loss.

    Total wall time includes setup, validation and parameter hashing. All GPU phase
    boundaries synchronize pending work. Memory peaks refer to this invocation.
    """
    if not calibration:
        validate_probe_splits(probe_splits)
    if calibration and (diagnostic or temporal):
        raise ValueError("Calibration-only omits probes; use diagnostic/temporal to score velocity.")
    start = time.perf_counter()
    device = resolve_device(device)
    _configure(config.seed)
    with _PeakMemory(device) as memory:
        model = Autoencoder(config.latent_dim, config.background).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, foreach=False)
        stream = MovieStream(np.random.SeedSequence([0, config.seed]), **config.movie_kwargs())
        model.train()
        _sync(device)
        train_start = time.perf_counter()
        for _ in range(config.steps):
            batch = stream.sample(config.batch_size)
            x = torch.from_numpy(batch.input).to(device)
            target = torch.from_numpy(batch.target).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(x), target, config.loss)
            loss.backward()
            optimizer.step()
        _sync(device)
        train_end = time.perf_counter()
        last_train_loss = loss.item()
        del optimizer, batch, x, target, loss
        if not np.isfinite(last_train_loss):
            raise FloatingPointError(f"Training ended with nonfinite loss {last_train_loss}")
        validation_loss = _validate(model, config, device)
        if not np.isfinite(validation_loss):
            raise FloatingPointError(f"Validation loss is nonfinite: {validation_loss}")
        _sync(device)
        evaluation_start = time.perf_counter()
        metrics, extra = {}, {}
        if diagnostic:
            from diagnostic_probes import evaluate_diagnostic
            extra = evaluate_diagnostic(model, config, device, probe_splits=probe_splits)
            metrics = extra.pop("direct")
        elif not calibration:
            metrics = evaluate(model, **config.movie_kwargs(), device=device, batch_size=config.batch_size,
                               n_train=config.probe_train, n_test=config.probe_test, temporal=temporal,
                               probe_splits=probe_splits, affine=affine)
            if temporal:
                extra["temporal_reconstruction"] = metrics.pop("temporal_reconstruction")
        if not calibration and (diagnostic or temporal or tuple(probe_splits) != (2, 3)):
            extra["probe_splits"] = list(probe_splits)
        if config.reject_reflections:
            extra["train_window_counts"] = {"proposed": stream.proposed, "accepted": stream.accepted}
        _sync(device)
        evaluation_end = time.perf_counter()
        model_hash = _parameter_digest(model)
        if store_weights is not None:
            store_weights(model)
    return {
        "device": str(device), "model_sha256": model_hash, "last_train_loss": last_train_loss,
        "validation_loss": validation_loss, "metrics": metrics, "memory": memory.result(), **extra,
        "timing": {"train_seconds": train_end - train_start,
                   "validation_seconds": evaluation_start - train_end,
                   "evaluation_seconds": evaluation_end - evaluation_start,
                   "total_seconds": time.perf_counter() - start},
    }
