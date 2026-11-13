"""Seeded two-frame movies. Coordinates are in pixels, with pixel centers at integers."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

SIZE = 64


def validate_frame_gap(frame_gap):
    if isinstance(frame_gap, bool) or not isinstance(frame_gap, (int, np.integer)) or frame_gap < 1:
        raise ValueError("frame_gap must be a positive integer number of frames")


def reflect(position, velocity, dt, radius):
    """Advance a disk center; velocity is the outgoing velocity at an exact collision."""
    lo, hi = radius - 0.5, SIZE - 0.5 - radius
    width = hi - lo
    phase = np.remainder(position - lo + dt * velocity, 2 * width)
    center = lo + np.minimum(phase, 2 * width - phase)
    speed = np.where(phase < width, velocity, -velocity)
    speed = np.where(phase == 0, np.abs(velocity), speed)
    speed = np.where(phase == width, -np.abs(velocity), speed)
    return center, speed


def disk_images(centers, radius, background=0.3, foreground=1.0):
    """Exact disk/pixel overlap by integrating one quadrant at pixel corners.

    For x,y >= 0 the cumulative area is the integral of
    min(y, sqrt(r²-u²)) from 0 to x (both limits clipped at r).
    Rectangle areas follow by differencing cumulative areas at four corners.
    Evaluate only the small bounding tile for each disk.
    """
    centers = np.asarray(centers, dtype=np.float64)
    flat = centers.reshape(-1, 2)
    side = 2 * radius + 2
    origin = np.clip(np.floor(flat).astype(np.int64) - radius, 0, SIZE - side)
    edges = origin[:, :, None] + np.arange(side + 1) - 0.5 - flat[:, :, None]
    xe, ye = edges[:, 0, :], edges[:, 1, :]
    x, y = np.minimum(abs(xe), radius), np.minimum(abs(ye), radius)
    crossing = np.sqrt(np.maximum(0, radius**2 - y**2))

    def arc_integral(u):
        return 0.5 * (u * np.sqrt(np.maximum(0, radius**2 - u**2)) +
                      radius**2 * np.arcsin(np.clip(u / radius, 0, 1)))

    area = np.where(x[:, None, :] <= crossing[:, :, None],
                    y[:, :, None] * x[:, None, :],
                    (y * crossing - arc_integral(crossing))[:, :, None] +
                    arc_integral(x)[:, None, :])
    area *= np.sign(ye)[:, :, None] * np.sign(xe)[:, None, :]
    coverage = np.clip(np.diff(np.diff(area, axis=2), axis=1), 0, 1)
    # Suppress roundoff for pixels completely outside or inside the disk.
    coverage[coverage < 1e-12] = 0.0
    coverage[coverage > 1 - 1e-12] = 1.0
    tile = (background + (foreground - background) * coverage).astype(np.float32)
    frames = np.full((len(flat), SIZE, SIZE), background, dtype=np.float32)
    pixels = origin[:, :, None] + np.arange(side)
    frames[np.arange(len(flat))[:, None, None], pixels[:, 1, :, None],
           pixels[:, 0, None, :]] = tile
    return frames.reshape(*centers.shape[:-1], SIZE, SIZE)


def noise(image, rng, kind, level):
    if level == 0:
        return image
    if kind == "gaussian":
        # Negative values and values above one intentionally survive.
        return image + np.float32(level) * rng.standard_normal(image.shape, dtype=np.float32)
    draw = rng.random(image.shape, dtype=np.float32)
    result = image.copy()
    result[draw < level / 2] = 0.0
    result[(draw >= level / 2) & (draw < level)] = 1.0
    return result


@dataclass
class Batch:
    input: np.ndarray
    target: Optional[np.ndarray]
    clean: np.ndarray
    labels: np.ndarray
    centers: np.ndarray


class MovieStream:
    def __init__(self, seed, *, radius, noise_type="gaussian", noise_level=0.0,
                 noise_placement="target_only", background=0.3, foreground=1.0, frame_gap=1):
        validate_frame_gap(frame_gap)
        self.frame_gap = int(frame_gap)
        if radius not in (1, 2, 3, 5, 8):
            raise ValueError("radius must be one of 1, 2, 3, 5, 8")
        if not (0 <= background <= 1 and 0 <= foreground <= 1):
            raise ValueError("foreground and background must be intensities in [0,1]")
        if noise_type not in ("gaussian", "salt_pepper"):
            raise ValueError("unsupported noise type")
        if not np.isfinite(noise_level) or noise_level < 0 or (noise_type == "salt_pepper" and noise_level > 1):
            raise ValueError("invalid noise level")
        if noise_placement not in ("target_only", "input_only", "both"):
            raise ValueError("unsupported noise placement")
        self.radius, self.background, self.foreground = radius, background, foreground
        self.placement, self.kind, self.level = noise_placement, noise_type, noise_level
        # Spawn independent streams without mutating a caller-supplied SeedSequence.
        seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        base = np.random.SeedSequence(seq.entropy, spawn_key=seq.spawn_key, pool_size=seq.pool_size)
        self.motion, self.input_rng, self.target_rng = map(np.random.default_rng, base.spawn(3))

    def sample(self, n, *, need_target=True):
        # Draw the four independent uniforms together so motion is invariant to chunk size.
        draw = self.motion.random((n, 4))
        previous = self.radius - 0.5 + (SIZE - 2 * self.radius) * draw[:, :2]
        velocity = 6 * draw[:, 2:] - 3
        current, outgoing = reflect(previous, velocity, 1, self.radius)
        future, _ = reflect(current, outgoing, 5, self.radius)
        # Anchor each gap to the same historical t-state so only the first
        # observed frame changes. Time-reverse the reflecting trajectory to
        # recover t-gap. Keep the original coordinates bit-for-bit for gap 1.
        if self.frame_gap != 1:
            previous, _ = reflect(current, -outgoing, self.frame_gap, self.radius)
        centers = np.stack((previous, current), axis=1)
        clean = disk_images(centers, self.radius, self.background, self.foreground)
        inputs = (noise(clean, self.input_rng, self.kind, self.level)
                  if self.placement in ("input_only", "both") else clean)
        target = None
        if need_target:
            target = (noise(clean, self.target_rng, self.kind, self.level)
                      if self.placement in ("target_only", "both") else clean)
        labels = np.concatenate((current, outgoing, future), axis=1).astype(np.float32)
        return Batch(inputs, target, clean, labels, centers.astype(np.float32))
