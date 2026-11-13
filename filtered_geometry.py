"""Measure window rejection, position/velocity bias, and six filtered true-center controls."""
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from data import MovieStream, reflection_axes
from diagnostic_probes import paired_scores
from evaluation import predict_probes

OUTPUT = Path(__file__).resolve().parent / "results" / "filtered_geometry.json"
SPLITS = {"fit": (2, 10000), "test": (3, 5000)}
NAMES = ["x", "y", "vx", "vy"]


def distribution(labels):
    xyv = labels[:, :4]
    correlation = np.corrcoef(xyv, rowvar=False)
    return {"n": len(xyv), "mean": xyv.mean(axis=0).tolist(),
            "sd": xyv.std(axis=0, ddof=1).tolist(), "correlations": correlation.tolist(),
            "position_velocity_correlations": correlation[:2, 2:].tolist()}


def condition(radius, gap):
    data, stats = {}, {}
    for name, (split, n) in SPLITS.items():
        kw = dict(radius=radius, frame_gap=gap)
        stream = MovieStream(np.random.SeedSequence([split, 0]), **kw, reject_reflections=True)
        centers, labels = stream.geometry(n)
        stats[name] = dict(accepted=n, proposed=stream.proposed, rejected=stream.proposed-n,
                           rejection_fraction=(stream.proposed-n)/stream.proposed)
        # Return the actual prefix encountered before the nth acceptance, including rejections.
        prop = MovieStream(np.random.SeedSequence([split, 0]), **kw)
        _, proposed_labels = prop.geometry(stream.proposed)
        keep = ~reflection_axes(proposed_labels[:, :2], proposed_labels[:, 2:4], radius, gap).any(axis=1)
        np.testing.assert_array_equal(proposed_labels[keep], labels)
        stats[name]["proposal_distribution"] = distribution(proposed_labels)
        stats[name]["accepted_distribution"] = distribution(labels)
        # All observed displacements recover instantaneous velocity before the first impact.
        difference = (centers[:, 1] - centers[:, 0]) / gap
        np.testing.assert_allclose(difference, labels[:, 2:4], rtol=0., atol=1e-12)
        assert not reflection_axes(centers[:, 1], labels[:, 2:4], radius, gap).any()
        future_hits = ((centers[:, 1] + 5*labels[:, 2:4] <= radius-.5) |
                       (centers[:, 1] + 5*labels[:, 2:4] >= 63.5-radius)).any(axis=1)
        stats[name]["future_reflecting"] = int(future_hits.sum())
        data[name] = centers.reshape(n, 4).astype(np.float32), labels[:, 2:4].astype(np.float32)
    result = {"radius": radius, "frame_gap": gap, "splits": stats, "oracle": None}
    if gap in (2, 4):
        x, y = data["fit"]
        xt, yt = data["test"]
        predictions = predict_probes(x, y, xt)
        oracle = {k: paired_scores(yt, p) for k, p in predictions.items()}
        oracle["passed"] = oracle["mlp"]["r2"] >= .95
        result["oracle"] = oracle
    return result


def main():
    rows = []
    for gap in (1, 2, 4, 8):
        for radius in (2, 5, 8):
            row = condition(radius, gap)
            rows.append(row)
            print(f"gap={gap} radius={radius} selected test prefix {row['splits']['test']['proposed']}, "
                  f"oracle={row['oracle']['mlp']['r2'] if row['oracle'] else 'unmeasured'}", flush=True)
    six = [r["oracle"] for r in rows if r["oracle"] is not None]
    payload = {"protocol": "observed-no-reflection-v1", "reject_reflections": True,
               "seed_linkage": "All AE seeds share these fixed probe streams; one oracle per (radius,gap).",
               "fit": {"n": 10000, "seed_sequence": [2,0]}, "test": {"n":5000,"seed_sequence":[3,0]},
               "geometry_precision": "float64; probe coordinates and labels match float32 training arrays",
               "distribution_columns": NAMES, "distributions": "sample SD (ddof=1), Pearson r",
               "regression": "true endpoints in chronological order to vx,vy. Fixed MLP 64x64, seeds 107/108, Adam 0.001, 80 epochs, batch 256. Ridge alpha=1.",
               "runtime": {"numpy": np.__version__, "torch": str(torch.__version__), "device": "cpu"},
               "all_six_oracles_passed": len(six) == 6 and all(o["passed"] for o in six),
               "conditions": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=OUTPUT.parent, delete=False) as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")
    Path(file.name).replace(OUTPUT)


if __name__ == "__main__":
    main()
