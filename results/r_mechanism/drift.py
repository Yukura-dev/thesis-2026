"""D addendum: does z keep changing during the stall, beyond a per-dimension shift and scale?

For every base run and every pair of checkpoints (i, j) on the split-3 test set:
- cka: linear CKA between z_i and z_j (invariant to rotation and isotropic scale, not to anisotropic scale);
- affine_r2: share of z_j's variance explained by an affine map of z_i (1 = z_j is an affine image of z_i);
- transfer_restd: velocity ridge probe trained at i (split 2), applied at j after re-standardizing z_j with
  split-2 statistics of checkpoint j (tolerates per-dimension shift and scale, not rotation).
usage: python drift.py  (all base runs)
"""
import numpy as np
import torch

from mech import OUT, STEPS, dump, load, r2, sample

dev = torch.device("cuda")


def enc(model, x):
    with torch.no_grad():
        return np.concatenate([model.encode(torch.from_numpy(x[i:i + 1024]).to(dev)).double().cpu().numpy()
                               for i in range(0, len(x), 1024)])


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def affine_r2(a, b):
    A = np.c_[a, np.ones(len(a))]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(1 - np.square(b - A @ coef).sum() / np.square(b - b.mean(0)).sum())


rows = []
for arm in ("l1", "wl1", "gauss", "l2"):
    for R in (5, 8):
        for S in range(5):
            cfg, _ = load(arm, R, S, 0, dev)
            xtr, _, vtr = sample(cfg, 2, 4096)
            xte, _, vte = sample(cfg, 3, 2048)
            Z, W = {}, {}
            for st in STEPS:
                _, model = load(arm, R, S, st, dev)
                ztr, zte = enc(model, xtr), enc(model, xte)
                m, s = ztr.mean(0), ztr.std(0)
                s[s < 1e-12] = 1
                zs = (ztr - m) / s
                W[st] = np.linalg.solve(zs.T @ zs + np.eye(16), zs.T @ (vtr - vtr.mean(0))), vtr.mean(0)
                Z[st] = (zte - m) / s
            for i in STEPS:
                for j in STEPS:
                    if j < i:
                        continue
                    w, ym = W[i]
                    rows.append({"arm": arm, "radius": R, "seed": S, "i": i, "j": j, "cka": cka(Z[i], Z[j]),
                                 "affine_r2": affine_r2(Z[i], Z[j]), "transfer_restd": r2(vte, Z[j] @ w + ym)})
            print(arm, R, S, flush=True)
dump(OUT / "drift.jsonl", rows)
