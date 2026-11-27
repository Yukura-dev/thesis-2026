"""Follow-up F1: is early encoder learning shared across losses?

Same seed (same initialization and same training batches), same step, different loss: linear CKA and affine R²
between the encoders' representations (z and the last conv block enc4, 512-d) on the split-3 test set.
Also each arm's enc4 representation at step s against its own step-0 representation.
"""
import numpy as np
import torch

from mech import OUT, dump, load, sample

dev = torch.device("cuda")
ARMS = ("l1", "wl1", "gauss", "l2")
STEPS = (1, 2, 5, 10, 25, 50, 100, 300, 1000, 3000)


def reps(model, x):
    acts = []
    h = model.ae.encoder[11].register_forward_hook(lambda m, i, o: acts.append(o.flatten(1).double().cpu()))
    zs = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            zs.append(model.encode(torch.from_numpy(x[i:i + 1024]).to(dev)).double().cpu())
    h.remove()
    return {"z": torch.cat(zs).numpy(), "enc4": torch.cat(acts).numpy()}


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    # kernel form: works for wide matrices
    ka, kb = a @ a.T, b @ b.T
    return float((ka * kb).sum() / np.sqrt((ka * ka).sum() * (kb * kb).sum()))


rows = []
for R in (5, 8):
    for S in range(5):
        cfg, _ = load("l1", R, S, 0, dev)
        x, _, _ = sample(cfg, 3, 1024)
        init = reps(load("l1", R, S, 0, dev)[1], x)
        for st in STEPS:
            rp = {a: reps(load(a, R, S, st, dev)[1], x) for a in ARMS}
            for k in ("z", "enc4"):
                for i, a in enumerate(ARMS):
                    rows.append({"radius": R, "seed": S, "step": st, "rep": k, "a": a, "b": "init", "cka": cka(rp[a][k], init[k])})
                    for b in ARMS[i + 1:]:
                        rows.append({"radius": R, "seed": S, "step": st, "rep": k, "a": a, "b": b, "cka": cka(rp[a][k], rp[b][k])})
        print(R, S, flush=True)
dump(OUT / "cross_arm.jsonl", rows)
