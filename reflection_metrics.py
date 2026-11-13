"""Reflection labels, true-center oracle readouts, and Bayes limits (original sampler)."""
import argparse
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from data import MovieStream, SIZE, reflect
from diagnostic_probes import paired_scores
from evaluation import predict_probes

OUTPUT = Path(__file__).resolve().parent / "results" / "reflection_oracle.json"


def reflection_axes(current, outgoing, radius, frame_gap):
    """Whether either center reaches a wall in (t−gap,t], by axis in reverse time.

    On an exact wall at t the outgoing (forward-time) sign points inward, so
    backtracking goes out immediately. If the first wall is reached exactly
    at t−gap, it is outside this half-open interval.
    """
    lo, hi = radius - .5, SIZE - .5 - radius
    unfurled = np.asarray(current, np.float64) - frame_gap * np.asarray(outgoing, np.float64)
    return (unfurled < lo) | (unfurled > hi)


def historical_geometry(radius, frame_gap, split=3, count=5000):
    """Recover original clean centers and collision labels *before* float32 conversion.

    Replay only the four independent uniforms and the published reflection law.
    The oracle verifies these coordinates and velocities against real MovieStream
    outputs on every train/test batch before scoring.
    """
    rng = MovieStream(np.random.SeedSequence([split, 0]), radius=radius).motion
    uniform = rng.random((count, 4))
    anchor = radius - .5 + (SIZE - 2 * radius) * uniform[:, :2]
    current, velocity = reflect(anchor, 6. * uniform[:, 2:] - 3., 1, radius)
    previous = anchor if frame_gap == 1 else reflect(current, -velocity, frame_gap, radius)[0]
    return {"centers": np.stack((previous, current), axis=1), "velocity": velocity,
            "hit_axes": reflection_axes(current, velocity, radius, frame_gap)}


def stratified_velocity(truth, prediction, hit_axes):
    """Scores on fixed subgroups. R² is centered within each scored group."""
    hit = np.asarray(hit_axes, bool).any(axis=1)
    result = {}
    for name, mask in (("all", np.ones(len(hit), bool)), ("reflecting", hit), ("nonreflecting", ~hit)):
        count = int(mask.sum())
        scores = paired_scores(truth[mask], prediction[mask]) if count >= 2 else {}
        result[name] = {"n": count, **scores}
    return result


def bayes_endpoints(centers, radius, frame_gap):
    """Conditional mean outgoing speed and candidate count under the unfiltered prior.

    For an axis of width W, the maximal path length L=3*gap is at most W/2 on
    this grid. Feasible paths are direct, one left hit, or one right hit, each
    mapping uniform initial speed to the observed endpoint with Jacobian Δ.
    The position at t-gap is uniform and independent of initial velocity in
    the stationary reflecting process, hence feasible candidates have equal
    posterior weight. This formula assumes ordinary interior endpoints;
    pairs exactly at a wall have probability zero under the continuous prior.
    """
    width = SIZE - 2 * radius
    length = 3 * frame_gap
    if length > width / 2:
        raise ValueError("Bayes implementation is valid only for 3*gap <= axis width/2")
    x, y = np.moveaxis(np.asarray(centers, np.float64) - (radius - .5), 1, 0)
    lower = (x + y <= length)
    upper = (2 * width - x - y <= length)
    direct_speed = (y - x) / frame_gap
    candidates = 1 + lower.astype(int) + upper.astype(int)
    prediction = (direct_speed + lower*(x+y)/frame_gap -
                  upper*(2*width-x-y)/frame_gap) / candidates
    return prediction, candidates


def _verified_geometry(radius, frame_gap, split, count):
    geometry = historical_geometry(radius, frame_gap, split, count)
    stream = MovieStream(np.random.SeedSequence([split, 0]), radius=radius, frame_gap=frame_gap)
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        batch = stream.sample(stop - start, need_target=False)
        np.testing.assert_array_equal(geometry["centers"][start:stop].astype(np.float32), batch.centers)
        np.testing.assert_array_equal(geometry["velocity"][start:stop].astype(np.float32), batch.labels[:, 2:4])
    return geometry


def oracle(radius, frame_gap):
    train = _verified_geometry(radius, frame_gap, 2, 10000)
    test = _verified_geometry(radius, frame_gap, 3, 5000)
    # Match label precision and all MLP optimizer / standardization conventions
    # of the six-target z probe, while fitting only vx,vy from four centers.
    fit_centers = train["centers"].reshape(-1, 4).astype(np.float32)
    test_centers = test["centers"].reshape(-1, 4).astype(np.float32)
    targets = test["velocity"].astype(np.float32)
    predictions = predict_probes(fit_centers, train["velocity"].astype(np.float32), test_centers)
    bayes, candidates = bayes_endpoints(test["centers"], radius, frame_gap)
    hit = test["hit_axes"]
    ambiguous = (candidates > 1).any(axis=1)
    # For endpoints in the ambiguous triangle beside a wall, conditional
    # variance = x²/Δ² (x is the old center's wall distance) and density =
    # 1/(3*Δ*W). Integrating over both triangles gives MSE=1.5*3*Δ/W;
    # dividing by Var(v)=3 yields R²=1−3*Δ/(2W).
    width = SIZE - 2*radius
    return {"radius": radius, "frame_gap": frame_gap,
            "counts": {"total": len(hit), "reflecting": int(hit.any(axis=1).sum()),
                       "nonreflecting": int((~hit.any(axis=1)).sum()),
                       "ambiguous_from_centers": int(ambiguous.sum())},
            "oracle_mlp": stratified_velocity(targets, predictions["mlp"], hit),
            "bayes_endpoint_mean": stratified_velocity(targets, bayes, hit),
            "population_bayes_r2": 1 - 3*frame_gap/(2*width)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metadata = {"design": "original reflecting sampler, Gaussian level 0, target_only",
                "inputs": "(x_old,y_old,x_t,y_t) true centers; outputs: outgoing (vx,vy) in px/frame",
                "seed_namespaces": {"fit": [2, 0], "test": [3, 0]},
                "fit_n": 10000, "test_n": 5000, "batch_size": 128, "mlp": {
                    "hidden_widths": [64,64], "activation": "ReLU", "optimizer": "Adam, lr=0.001",
                    "epochs": 80, "batch_size": 256, "initialization_seed": 107, "shuffle_seed": 108},
                "runtime": {"numpy": np.__version__, "torch": str(torch.__version__),
                            "python": platform.python_version(), "device": "cpu"},
                "subgroups": "any axis hits a wall in (t-frame_gap,t], neither mask nor velocity given to predictor",
                "population_bayes_formula": "1 - 3 * frame_gap / (2 * (64 - 2 * radius))",
                "conditions": []}
    for gap in (1, 2, 4, 8):
        for radius in (2, 5, 8):
            row = oracle(radius, gap)
            metadata["conditions"].append(row)
            print(f"gap={gap} r={radius}, true-center MLP={row['oracle_mlp']['all']['r2']:.6f}, "
                  f"test Bayes={row['bayes_endpoint_mean']['all']['r2']:.6f}, "
                  f"population Bayes={row['population_bayes_r2']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    Path(file.name).replace(args.output)


if __name__ == "__main__":
    main()
