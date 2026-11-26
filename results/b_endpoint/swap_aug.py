"""Swap-augmented velocity probe for the saved Experiment N models. See PLAN_swap_aug.md."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_endpoint import BASE, CACHE, CONDITIONS, fit_big, r2  # noqa: E402
from data import MovieStream  # noqa: E402
from evaluation import predict_probes  # noqa: E402
from training import Autoencoder, _parameter_digest, resolve_device  # noqa: E402

device = resolve_device("cuda")
original = {(r["condition"], r["seed"]): r for r in map(json.loads, (OUT / "runs.jsonl").read_text().splitlines())}


def both_orders(model, cfg, split, size):
    stream = MovieStream(np.random.SeedSequence([split, 0]), **cfg.movie_kwargs())
    zt, zs, vt = [], [], []
    with torch.inference_mode():
        for start in range(0, size, cfg.batch_size):
            b = stream.sample(min(cfg.batch_size, size - start), need_target=False)
            x = torch.from_numpy(b.input).to(device)
            zt.append(model.encode(x).cpu().numpy())
            zs.append(model.encode(x.flip(1)).cpu().numpy())
            vt.append(b.labels[:, 2:4].astype(np.float64))
    zt, zs, vt = map(np.concatenate, (zt, zs, vt))
    return np.concatenate((zt, zs)), np.concatenate((vt, -vt))


rows = []
for name, (overrides, init) in CONDITIONS.items():
    for seed in range(5):
        cfg = replace(BASE, seed=seed, **overrides)
        model = Autoencoder(cfg.latent_dim, cfg.background, initialization=init).to(device)
        model.load_state_dict(torch.load(CACHE / f"{name}_s{seed}.pt", map_location=device))
        model.eval()
        assert _parameter_digest(model) == original[(name, seed)]["model_sha256"]
        z, v = both_orders(model, cfg, 2, cfg.probe_train)
        zt, vt = both_orders(model, cfg, 3, 5000)
        row = {"condition": name, "seed": seed, "frames": original[(name, seed)]["frames"],
               "aug_v_r2_mlp": r2(vt, predict_probes(z, v, zt)["mlp"]),
               "aug_v_r2_big": r2(vt, fit_big(z, v, zt))}
        rows.append(row)
        print(name, seed, row["frames"], f"mlp={row["aug_v_r2_mlp"]:.3f}", f"big={row["aug_v_r2_big"]:.3f}", flush=True)
(OUT / "swap_aug.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
