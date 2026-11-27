"""Train the Experiment R runs with checkpoints; check base arms against the thesis ledger.

usage: python train.py ARM [ARM ...] [--radius 5 8] [--seeds 0 1 2 3 4]
"""
import argparse
import json
import time

import torch

from mech import BASE_ARMS, OUT, RADII, SEEDS, dump, ledger_shas, run_dir, train_run

p = argparse.ArgumentParser()
p.add_argument("arms", nargs="+")
p.add_argument("--radius", type=int, nargs="+", default=list(RADII))
p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
a = p.parse_args()
dev = torch.device("cuda")
shas = ledger_shas()
for arm in a.arms:
    for r in a.radius:
        for s in a.seeds:
            if (run_dir(arm, r, s) / "3000.pt").exists():
                continue
            t = time.time()
            cfg, model, sha = train_run(arm, r, s, dev)
            key = (r, cfg.loss, 0.0, cfg.loss_scale, s, "default")
            match = (sha in shas[key]) if (arm in BASE_ARMS and key in shas) else None
            row = {"arm": arm, "radius": r, "seed": s, "loss": cfg.loss, "loss_scale": cfg.loss_scale,
                   "model_sha256": sha, "ledger_match": match, "seconds": time.time() - t}
            dump(OUT / "train.jsonl", [row])
            print(json.dumps(row), flush=True)
            if arm in BASE_ARMS and match is not True:
                raise SystemExit(f"ledger mismatch for {arm} r{r} s{s}")
