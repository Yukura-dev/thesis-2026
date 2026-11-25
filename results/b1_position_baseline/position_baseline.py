"""B-1: how much velocity R² is reachable from true positions alone.

Uses the exact probe pipeline of evaluation.py (same splits 2/3, n=10000/5000,
same standardization, ridge alpha=1, same seeded 2x64 MLP) but feeds ground-truth
centers instead of AE latents. Main-experiment geometry: frame_gap=4,
reject_reflections=True. Labels: [current x,y | velocity vx,vy | future x,y].
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from data import MovieStream
from evaluation import predict_probes


def r2(truth, pred):
    res = np.square(truth - pred).sum(0)
    var = np.square(truth - truth.mean(0)).sum(0)
    return (1 - res / var).mean()


def geom(split, n, radius, reject, gap):
    s = MovieStream(np.random.SeedSequence([split, 0]), radius=radius,
                    frame_gap=gap, reject_reflections=reject)
    centers, labels = s.geometry(n)
    return centers, labels


rows = []
for reject in (True, False):
    for radius in (1, 2, 3, 5, 8):
        ctr, ytr = geom(2, 10000, radius, reject, 4)
        cte, yte = geom(3, 5000, radius, reject, 4)
        corr = np.corrcoef(ytr[:, 0], ytr[:, 2])[0, 1]
        feats = {"current": lambda c: c[:, 1], "previous": lambda c: c[:, 0],
                 "both": lambda c: c.reshape(len(c), 4)}
        for name, f in feats.items():
            p = predict_probes(f(ctr), ytr[:, 2:4], f(cte))
            row = {"reject": reject, "radius": radius, "input": name,
                   "vel_r2_mlp": round(float(r2(yte[:, 2:4], p["mlp"])), 4),
                   "vel_r2_ridge": round(float(r2(yte[:, 2:4], p["ridge"])), 4),
                   "corr_x_vx": round(float(corr), 3)}
            rows.append(row)
            print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
