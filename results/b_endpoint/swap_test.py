"""Frame-swap invariance and past-only oracle for the saved Experiment N models. See PLAN_swap.md."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_endpoint import BASE, CACHE, CONDITIONS, analyse, collect, r2, targets  # noqa: E402
from data import MovieStream  # noqa: E402
from evaluation import predict_probes  # noqa: E402
from training import Autoencoder, _parameter_digest, resolve_device  # noqa: E402

device = resolve_device("cuda")
original = {(r["condition"], r["seed"]): r for r in map(json.loads, (OUT / "runs.jsonl").read_text().splitlines())}

# Past-only oracle, per radius.
oracle = {}
for radius in (5, 8):
    cfg = replace(BASE, radius=radius)
    data = {}
    for split, size in ((2, 10000), (3, 5000)):
        c, lab = MovieStream(np.random.SeedSequence([split, 0]), **cfg.movie_kwargs()).geometry(size)
        data[split] = targets(np.asarray(c, np.float64), np.asarray(lab, np.float64))
    past = lambda t: t["ends"][:, :2]
    oracle[radius] = analyse(past(data[2]), data[2], past(data[3]), data[3], big=False)["mlp"]
(OUT / "references_past_only.json").write_text(json.dumps(oracle, indent=1) + "\n")
print("past-only oracle:", {r: (round(v["cur"], 4), round(v["v_direct"], 4)) for r, v in oracle.items()}, flush=True)

perm = np.random.default_rng(11).permutation(5000)
rows = []
for name, (overrides, init) in CONDITIONS.items():
    for seed in range(5):
        cfg = replace(BASE, seed=seed, **overrides)
        model = Autoencoder(cfg.latent_dim, cfg.background, initialization=init).to(device)
        model.load_state_dict(torch.load(CACHE / f"{name}_s{seed}.pt", map_location=device))
        model.eval()
        assert _parameter_digest(model) == original[(name, seed)]["model_sha256"]
        z, c, lab = collect(model, cfg, 2, cfg.probe_train, device)
        stream = MovieStream(np.random.SeedSequence([3, 0]), **cfg.movie_kwargs())
        zt, zs, vt = [], [], []
        with torch.inference_mode():
            for start in range(0, 5000, cfg.batch_size):
                b = stream.sample(min(cfg.batch_size, 5000 - start), need_target=False)
                x = torch.from_numpy(b.input).to(device)
                zt.append(model.encode(x).cpu().numpy())
                zs.append(model.encode(x.flip(1)).cpu().numpy())
                vt.append(b.labels[:, 2:4].astype(np.float64))
        zt, zs, vt = map(np.concatenate, (zt, zs, vt))
        ratio = float(np.square(zt - zs).sum(1).mean() / np.square(zt - zt[perm]).sum(1).mean())
        pred = predict_probes(z, lab[:, 2:4], np.concatenate((zt, zs)))["mlp"]
        row = {"condition": name, "seed": seed, "frames": original[(name, seed)]["frames"],
               "swap_ratio": ratio, "v_r2_original": r2(vt, pred[:5000]),
               "swap_consistency_r2": r2(-vt, pred[5000:])}
        rows.append(row)
        print(name, seed, row["frames"], f"R={ratio:.4f}", f"v={row['v_r2_original']:.3f}",
              f"swap_cons={row['swap_consistency_r2']:.3f}", flush=True)
(OUT / "swap_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
