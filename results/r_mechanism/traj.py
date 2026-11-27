"""D (trajectory), B/G5 (sym/asym) and A (layerwise) for one run. usage: python traj.py ARM RADIUS SEED"""
import sys
import time

import numpy as np
import torch

from mech import (OUT, STEPS, dump, fit_big, image_metrics, load, mlp, probe_all, r2, ridge, sample, targets)

arm, R, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
torch.set_num_threads(2)
LAYER_STEPS = (0, 25, 100, 3000)
cfg, _ = load(arm, R, S, 0, dev)
xtr, ctr, vtr = sample(cfg, 2, 4096)
xte, cte, vte = sample(cfg, 3, 2048)
xev, cev, _ = sample(cfg, 9, 1024)
ttr, tte = targets(ctr, vtr), targets(cte, vte)
perm = np.random.default_rng(11).permutation(len(xte))
import json  # noqa: E402


def _done(name):
    f = OUT / name
    return f.exists() and any((r["arm"], r["radius"], r["seed"]) == (arm, R, S) for r in map(json.loads, open(f)))


SKIP_D = _done("traj.jsonl")
if SKIP_D and (arm not in ("l1", "wl1", "gauss", "l2") or _done("layers.jsonl")):
    print("already done", arm, R, S)
    sys.exit(0)


def enc(model, x):
    with torch.no_grad():
        return np.concatenate([model.encode(torch.from_numpy(x[i:i + 1024]).to(dev)).double().cpu().numpy()
                               for i in range(0, len(x), 1024)])


def params(model):
    e, d = [], []
    for n, p in model.ae.named_parameters():
        (e if ("encoder" in n or "to_latent" in n) else d).append(p.detach().flatten())
    return torch.cat(e), torch.cat(d)


rows, prev_p, ridge_w = [], None, {}
t0 = time.time()
for step in (() if SKIP_D else STEPS):
    _, model = load(arm, R, S, step, dev)
    ztr, zte = enc(model, xtr), enc(model, xte)
    ztr_s, zte_s = enc(model, xtr[:, ::-1].copy()), enc(model, xte[:, ::-1].copy())
    with torch.no_grad():
        out = np.concatenate([model(torch.from_numpy(xev[i:i + 512]).to(dev)).cpu().numpy() for i in range(0, 1024, 512)])
    row = {"arm": arm, "radius": R, "seed": S, "step": step, "image": image_metrics(out, cev, R)}
    e, d = params(model)
    if prev_p is not None:
        row["rel_change_enc"] = float((e - prev_p[0]).norm() / prev_p[0].norm())
        row["rel_change_dec"] = float((d - prev_p[1]).norm() / prev_p[1].norm())
    row["dist_from_init_enc"] = float((e - first[0]).norm() / first[0].norm()) if prev_p is not None else 0.0
    row["dist_from_init_dec"] = float((d - first[1]).norm() / first[1].norm()) if prev_p is not None else 0.0
    if prev_p is None:
        first = (e, d)
    prev_p = (e, d)
    # latent statistics
    row["z_mean_norm"] = float(np.linalg.norm(zte.mean(0)))
    row["z_sd_total"] = float(np.sqrt(zte.var(0).sum()))
    # swap test (numerator, denominator in absolute terms)
    num = float(np.square(zte - zte_s).sum(1).mean())
    den = float(np.square(zte - zte[perm]).sum(1).mean())
    sym_te, asym_te = (zte + zte_s) / 2, (zte - zte_s) / 2
    sym_tr, asym_tr = (ztr + ztr_s) / 2, (ztr - ztr_s) / 2
    row.update(swap_num=num, swap_den=den, swap_ratio=num / den,
               asym_norm=float(np.sqrt(np.square(asym_te).sum(1).mean())),
               sym_norm=float(np.sqrt(np.square(sym_te - sym_te.mean(0)).sum(1).mean())),
               asym_var=float(asym_te.var(0).sum()), sym_var=float(sym_te.var(0).sum()))
    # probes on z
    row["probe_z"] = probe_all(ztr, ttr, zte, tte)
    kinds = ("ridge", "mlp")
    row["probe_sym"] = probe_all(sym_tr, ttr, sym_te, tte, kinds, ("mid", "vel"))
    row["probe_asym"] = probe_all(asym_tr, ttr, asym_te, tte, kinds, ("mid", "vel"))
    if step == 3000:
        row["probe_big"] = {"z_vel": r2(vte, fit_big(ztr, vtr, zte)), "asym_vel": r2(vte, fit_big(asym_tr, vtr, asym_te))}
    # cross-checkpoint linear transfer: keep the ridge map in raw z units
    m, s = ztr.mean(0), ztr.std(0)
    s[s < 1e-12] = 1
    zs = (ztr - m) / s
    w = np.linalg.solve(zs.T @ zs + np.eye(16), zs.T @ (vtr - vtr.mean(0)))
    ridge_w[step] = (m, s, w, vtr.mean(0))
    row["_zte"] = zte
    rows.append(row)
    print(f"{arm} r{R} s{S} step {step} {time.time() - t0:.0f}s", flush=True)

# transfer matrix: probe trained at i, applied to raw z at j
for row in rows:
    zte = row.pop("_zte")
    row["transfer_vel_r2"] = {str(i): r2(vte, ((zte - m) / s) @ w + ym) for i, (m, s, w, ym) in ridge_w.items()}
dump(OUT / "traj.jsonl", rows)

# A: layerwise probes
if arm in ("l1", "wl1", "gauss", "l2"):
    keys = ("prev", "cur", "mid", "vel", "dir")
    ytr = np.concatenate([ttr[k] for k in keys], 1)

    LAYERS = ("input", "enc1", "enc2", "enc3", "enc4", "z", "from_latent", "dec1", "dec2", "dec3", "output")

    def feats(model, x, layer):
        """Activations of one layer only (keeps memory to one feature matrix)."""
        if layer == "input":
            return torch.from_numpy(x.reshape(len(x), -1))
        mods = {"enc1": model.ae.encoder[2], "enc2": model.ae.encoder[5], "enc3": model.ae.encoder[8],
                "enc4": model.ae.encoder[11], "from_latent": model.ae.from_latent, "dec1": model.ae.decoder[2],
                "dec2": model.ae.decoder[5], "dec3": model.ae.decoder[8]}
        acts = []
        h = mods[layer].register_forward_hook(lambda m, i, o: acts.append(o.flatten(1).cpu())) if layer in mods else None
        with torch.no_grad():
            for i in range(0, len(x), 512):
                z = model.encode(torch.from_numpy(x[i:i + 512]).to(dev))
                if layer == "z":
                    acts.append(z.cpu())
                    continue
                out = model.decode(z)
                if layer == "output":
                    acts.append(out.flatten(1).cpu())
        if h is not None:
            h.remove()
        return torch.cat(acts)

    def ridge_dual(F, Ft, y, alpha):
        """Ridge on centred features (no per-feature scaling: near-constant pixels would blow up), float64.
        alpha is relative: the penalty is alpha * mean feature variance * n_features."""
        F, Ft, y = F.double(), Ft.double(), y.double()
        m = F.mean(0)
        F, Ft = F - m, Ft - m
        lam = alpha * float(F.square().sum()) / len(F)
        ym = y.mean(0)
        if F.shape[1] <= F.shape[0]:
            w = torch.linalg.solve(F.T @ F + lam * torch.eye(F.shape[1], device=dev, dtype=F.dtype), F.T @ (y - ym))
            return (Ft @ w + ym).float()
        K = F @ F.T
        coef = torch.linalg.solve(K + lam * torch.eye(len(F), device=dev, dtype=F.dtype), y - ym)
        return (Ft @ (F.T @ coef) + ym).float()

    def score(pred, yt):
        pred = pred.cpu().numpy() * y_sd + y_mu
        return {k: r2(yt[k], pred[:, 2 * i:2 * i + 2]) for i, k in enumerate(keys)}

    lrows = []
    y_mu, y_sd = ytr.mean(0), ytr.std(0)
    Y = torch.from_numpy((ytr - y_mu) / y_sd).float().to(dev)
    for step in LAYER_STEPS:
        _, model = load(arm, R, S, step, dev)
        for layer in LAYERS:
            F = feats(model, xtr, layer).float().to(dev)
            Ft = feats(model, xte, layer).float().to(dev)
            n = int(.8 * len(F))
            alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
            errs = [float(((ridge_dual(F[:n], F[n:], Y[:n], al) - Y[n:]) ** 2).mean()) for al in alphas]
            best = alphas[int(np.argmin(errs))]
            entry = {"arm": arm, "radius": R, "seed": S, "step": step, "layer": layer, "dim": F.shape[1],
                     "alpha": best, "ridge_full": score(ridge_dual(F, Ft, Y, best), tte)}
            Fc = F - F.mean(0)
            # Top principal axes from the smaller of the covariance and Gram eigenproblems (float64).
            Fd = Fc.double()
            if Fd.shape[1] <= Fd.shape[0]:
                lam, vec = torch.linalg.eigh(Fd.T @ Fd)
                lam, vec = lam.flip(0).clamp_min(0), vec.flip(1)
                V = vec.T.float()
            else:
                lam, vec = torch.linalg.eigh(Fd @ Fd.T)
                lam, vec = lam.flip(0).clamp_min(0), vec.flip(1)
                V = (Fd.T @ vec[:, :64] / lam[:64].clamp_min(1e-30).sqrt()).T.float()
            entry["pca_var64"] = float(lam[:64].sum() / lam.sum())
            for k in (16, 64):
                k2 = min(k, V.shape[0], int((lam > 1e-6 * lam[0]).sum()))  # drop numerically null axes
                P, Pt = Fc @ V[:k2].T, (Ft - F.mean(0)) @ V[:k2].T
                entry[f"ridge_pca{k}"] = score(ridge_dual(P, Pt, Y, 1e-3), tte)
                if k == 64:
                    entry["mlp_pca64"] = score(torch.from_numpy(mlp(P.cpu().numpy(), (ytr - y_mu) / y_sd, Pt.cpu().numpy())), tte)
            lrows.append(entry)
        print(f"layers {arm} r{R} s{S} step {step} {time.time() - t0:.0f}s", flush=True)
    dump(OUT / "layers.jsonl", lrows)
print("done", arm, R, S, f"{time.time() - t0:.0f}s", flush=True)
