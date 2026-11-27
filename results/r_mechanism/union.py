"""Follow-up F3 (exploratory, from the trajectory figure): is an order-free 'union' output a transient for every loss?

At every checkpoint, decode the split-9 set and measure, per output channel k, the mean brightness above
background on the channel's own disk (own) and on the other frame's disk (other), in units of fg - bg = 0.7.
union = both high and similar; ordered = own high, other low; flat = both low.
"""
import numpy as np
import torch

from mech import OUT, STEPS, disk_images, dump, load, sample

dev = torch.device("cuda")
rows = []
for arm in ("l1", "wl1", "gauss", "l2", "wl1_ord", "l1_bn"):
    for R in (5, 8):
        for S in range(5):
            cfg, _ = load(arm, R, S, 0, dev)
            x, c, _ = sample(cfg, 9, 512)
            cov = disk_images(c, R, 0, 1) > .5
            # the other frame's disk minus any overlap with the own disk
            other_only = [cov[:, 1 - k] & ~cov[:, k] for k in (0, 1)]
            own_only = [cov[:, k] & ~cov[:, 1 - k] for k in (0, 1)]
            for st in STEPS:
                _, model = load(arm, R, S, st, dev)
                with torch.no_grad():
                    out = model(torch.from_numpy(x).to(dev)).cpu().numpy() - .3
                own = np.mean([(out[:, k] * own_only[k]).sum() / own_only[k].sum() for k in (0, 1)]) / .7
                oth = np.mean([(out[:, k] * other_only[k]).sum() / other_only[k].sum() for k in (0, 1)]) / .7
                rows.append({"arm": arm, "radius": R, "seed": S, "step": st, "own": float(own), "other": float(oth)})
            print(arm, R, S, flush=True)
dump(OUT / "union.jsonl", rows)
