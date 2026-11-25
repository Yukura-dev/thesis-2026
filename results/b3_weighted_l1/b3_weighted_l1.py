"""B-3: is the L1 background trap a path effect of the first-order (kink) cost on background pixels?

Loss: plain L1 (no smoothing) with weight w on background pixels, 1 on object pixels.
Same protocol as experiment M (r=5,8, seeds 0-4, Delta=4, lr .003, 3000 steps, clean
input and target, probe splits 6/7, centre-intensity classification at 0.65).

Pre-registered predictions (written before running):
  * Constant-output optimum. A z-independent output minimises the weighted L1 pixel-wise
    at the weighted median. It stays at the background value iff w*(1-q_p) >= q_p for every
    pixel, q_p = probability that pixel p lies on the disk. So for w > w_opt = max q/(1-q)
    the trivial optimum is unchanged (computed below from the data stream).
  * Kink-cost hypothesis. From the background output, a localized change gains on the
    disk pixels and pays w * (spill on background pixels) at first order. Lowering w
    therefore removes the trap at some w_escape that can lie ABOVE w_opt.
  Outcome A (supports): velocity/both-frame success appears at some w > w_opt.
  Outcome B (against): success only appears for w <= w_opt, i.e. only where the trivial
    optimum itself has moved; then the trap is not separable from the optimum by this test.
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from data import MovieStream, disk_images
from training import Config, run_experiment

OUT = Path("results/b3_weighted_l1")
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "runs.jsonl"
WEIGHTS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01)
RADII, SEEDS, SPLITS, THRESHOLD = (5, 8), range(5), (6, 7), .65


def coverage_threshold(radius, n=20000):
    s = MovieStream(np.random.SeedSequence([0, 99]), radius=radius, frame_gap=4, reject_reflections=True)
    centers, _ = s.geometry(n)
    on = (disk_images(centers, radius, 0., 1.) > 0).mean(axis=0)  # (2, 64, 64) per frame
    q = on.max()
    return float(q), float(q / (1 - q))


def classify(m):
    p, c = m["center_intensity_previous"] >= THRESHOLD, m["center_intensity_current"] >= THRESHOLD
    return {(False, False): "background", (True, False): "past", (False, True): "current",
            (True, True): "both"}[(p, c)]


done = set()
if RUNS.exists():
    done = {(r["radius"], r["w"], r["seed"]) for r in map(json.loads, RUNS.read_text().splitlines())}
thr = {r: coverage_threshold(r) for r in RADII}
(OUT / "threshold.json").write_text(json.dumps({str(r): {"q_max": q, "w_opt": w} for r, (q, w) in thr.items()}, indent=1))
print("w_opt", {r: round(w, 4) for r, (_, w) in thr.items()}, flush=True)
base = Config(frame_gap=4, steps=3000, lr=.003, reject_reflections=True)
for r in RADII:
    for w in WEIGHTS:
        for seed in SEEDS:
            if (r, w, seed) in done:
                continue
            cfg = replace(base, radius=r, seed=seed,
                          **({"loss": "l1"} if w == 1.0 else {"loss": "bg_weighted_l1", "loss_scale": w}))
            t = time.time()
            res = run_experiment(cfg, "cuda", probe_splits=SPLITS, affine=True)
            m = res["metrics"]
            row = {"radius": r, "w": w, "seed": seed, "class": classify(m),
                   "velocity_r2_mlp": m["velocity_r2_mlp"], "position_r2_mlp": m["position_r2_mlp"],
                   "psnr": m["psnr"], "center_previous": m["center_intensity_previous"],
                   "center_current": m["center_intensity_current"], "validation_loss": res["validation_loss"],
                   "model_sha256": res["model_sha256"], "seconds": time.time() - t}
            with RUNS.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k not in ("model_sha256",)}), flush=True)
