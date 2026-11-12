"""Measure CPU batch generation for stage runtime estimates. Does not train a model."""
import json
from pathlib import Path
from random import Random
from time import perf_counter

from data import MovieStream
from run_sweep import LEVELS, RADII

OUTPUT = Path(__file__).resolve().parent / "results" / "generation_times.json"


def measure():
    cells = {(r, "gaussian", 0.0, "target_only") for r in RADII}
    for radius in RADII:
        for kind, levels in LEVELS.items():
            cells.update((radius, kind, level, "target_only") for level in levels if level > 0)
    for kind in LEVELS:
        cells.update((5, kind, 0.2, placement) for placement in ("input_only", "both"))
    observations = {cell: [] for cell in sorted(cells)}
    order = list(observations) * 3
    Random(1003).shuffle(order)
    for cell in order:
        radius, kind, level, placement = cell
        stream = MovieStream(974, radius=radius, noise_type=kind, noise_level=level,
                             noise_placement=placement)
        batch = stream.sample(128)
        start = perf_counter()
        for _ in range(8):
            batch = stream.sample(128)
        observations[cell].append((perf_counter() - start) / 8)
        del batch
    return {"method": "CPU only, batch size 128, 3 groups of 8 batches, one warmup per group.",
            "data_seed": 974, "order_seed": 1003,
            "cells": [{"radius": r, "noise_type": kind, "noise_level": level,
                       "placement": placement, "seconds_per_batch": samples}
                      for (r, kind, level, placement), samples in observations.items()]}


if __name__ == "__main__":
    result = measure()
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['cells'])} CPU generator timings to {OUTPUT}")
