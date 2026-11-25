"""Bayes-optimal R² of velocity from position alone, by fine binning of a large sample.
Axes are independent after rejection (accept = neither axis reflects), so per-axis 1-D
conditional means suffice. R² = Var(E[v|x]) / Var(v), averaged over the two axes."""
import sys, json
import numpy as np
sys.path.insert(0, ".")
from data import MovieStream

out = {}
for radius in (1, 2, 3, 5, 8):
    s = MovieStream(np.random.SeedSequence([12345, radius]), radius=radius, frame_gap=4, reject_reflections=True)
    centers, labels = s.geometry(4_000_000)
    v = labels[:, 2:4]
    res = {}
    for name, pos in (("current", centers[:, 1]), ("previous", centers[:, 0])):
        r2 = []
        for a in range(2):
            x, y = pos[:, a], v[:, a]
            bins = np.linspace(x.min(), x.max(), 801)
            idx = np.clip(np.digitize(x, bins) - 1, 0, 799)
            cnt = np.bincount(idx, minlength=800)
            mean = np.bincount(idx, y, minlength=800) / np.maximum(cnt, 1)
            r2.append(float(np.sum(cnt * (mean - y.mean()) ** 2) / np.sum((y - y.mean()) ** 2)))
        res[name] = round(float(np.mean(r2)), 4)
    out[radius] = res
    print(radius, res, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
