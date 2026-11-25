"""Retrain a few B-3 models (deterministic, same config) and show their reconstructions."""
import sys
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from data import MovieStream
from training import Config, run_experiment

out = sys.argv[1]
cases = [(5, 1.0), (5, 0.2), (5, 0.1), (5, 0.01), (5, "g0.4"), (8, 0.3), (8, 0.01), (8, "g0.1")]
base = Config(frame_gap=4, steps=3000, lr=.003, reject_reflections=True)
rows = []
for r, w in cases:
    if isinstance(w, str):
        cfg = replace(base, radius=r, seed=0, loss="gauss_l1", loss_scale=float(w[1:]))
        name = f"gauss_l1\nσ={w[1:]}"
    else:
        cfg = replace(base, radius=r, seed=0, **({"loss": "l1"} if w == 1 else {"loss": "bg_weighted_l1", "loss_scale": w}))
        name = "L1" if w == 1 else f"weighted L1\nw={w}"
    holder = {}
    res = run_experiment(cfg, "cuda", calibration=True, store_weights=lambda m: holder.setdefault("m", m))
    model = holder["m"].eval()
    batch = MovieStream(np.random.SeedSequence([7, 0]), **cfg.movie_kwargs()).sample(1, need_target=False)
    with torch.inference_mode():
        rec = model(torch.from_numpy(batch.input).cuda()).cpu().numpy()[0]
    rows.append((r, name, batch.input[0], rec))
    print(r, w, res["model_sha256"][:12], flush=True)

fig, ax = plt.subplots(4, 6, figsize=(10.5, 7.6), gridspec_kw=dict(wspace=0.05, hspace=0.3))
for block, rad in enumerate((5, 8)):
    grp = [x for x in rows if x[0] == rad]
    cols = [("input", grp[0][2])] + [(n, y) for _, n, _, y in grp]
    top = 2 * block
    for j in range(6):
        for f in range(2):
            a = ax[top + f, j]
            a.set_xticks([]); a.set_yticks([])
            if j < len(cols):
                a.imshow(cols[j][1][f], cmap="gray", vmin=0, vmax=1)
                if f == 0:
                    a.set_title(cols[j][0], fontsize=9)
            else:
                a.axis("off")
    ax[top, 0].set_ylabel(f"r={rad}\nt−4", rotation=0, ha="right", va="center", fontsize=10)
    ax[top + 1, 0].set_ylabel("t", rotation=0, ha="right", va="center", fontsize=10)
fig.savefig(out, dpi=170, bbox_inches="tight")
