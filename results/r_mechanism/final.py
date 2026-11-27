"""G2 (latent and decoder-entry statistics, every checkpoint) and E (interventions, step 3000) for one run.

usage: python final.py ARM RADIUS SEED
"""
import sys
import time

import numpy as np
import torch
from torch.func import jacfwd, vmap

from mech import OUT, STEPS, disk_images, dump, image_metrics, load, ridge, sample

arm, R, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
torch.set_num_threads(2)
cfg, _ = load(arm, R, S, 0, dev)
x4, c4, v4 = sample(cfg, 4, 4096)
x9, c9, v9 = sample(cfg, 9, 1024)
t0 = time.time()


def T(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)


@torch.no_grad()
def enc(model, x):
    return torch.cat([model.encode(T(x[i:i + 1024])) for i in range(0, len(x), 1024)])


@torch.no_grad()
def dec(model, z):
    return torch.cat([model.decode(z[i:i + 512]) for i in range(0, len(z), 512)]).cpu().numpy()


def decoder_entry(model, z):
    """Split the decoder's first activations into a constant part and an input-dependent part."""
    with torch.no_grad():
        zz = (z - model.shift) / model.scale
        if model.bn is not None:
            zz = model.bn(zz)
        h = model.ae.from_latent(zz)
        g = model.ae.decoder[1](model.ae.decoder[0](h.reshape(-1, 32, 4, 4))).flatten(1)  # after first GN
    out = {}
    for name, a in (("from_latent", h), ("dec_gn1", g)):
        const = a.mean(0)
        var = a - const
        out[f"{name}_const_rms"] = float(const.square().mean().sqrt())
        out[f"{name}_var_rms"] = float(var.square().mean().sqrt())
        out[f"{name}_var_frac"] = float(var.square().mean() / a.square().mean())
    return out


rows = []
for step in STEPS:
    _, model = load(arm, R, S, step, dev)
    z = enc(model, x4)
    zc = (z - z.mean(0)).double()
    lam = torch.linalg.eigvalsh(torch.cov(zc.T)).clamp_min(0).flip(0)
    row = {"arm": arm, "radius": R, "seed": S, "step": step, "z_mean_norm": float(z.mean(0).norm()),
           "z_var_rms": float(zc.square().sum(1).mean().sqrt()), "eig": lam.cpu().tolist(),
           "cond": float(lam[0] / lam[-1].clamp_min(1e-30)), "part_ratio": float(lam.sum() ** 2 / lam.square().sum()),
           **decoder_entry(model, z)}
    # decoder Jacobian dD/dz on 64 samples; energy along the velocity-readout directions vs all directions
    zs = z[:64].detach()
    J = vmap(jacfwd(lambda q: model.decode(q[None])[0].flatten()))(zs)  # (64, 8192, 16)
    fro2 = J.square().sum((1, 2)).mean()
    row["jac_fro"] = float(fro2.sqrt())
    zn = z.double().cpu().numpy()
    m, s = zn.mean(0), zn.std(0)
    s[s < 1e-12] = 1
    zs_ = (zn - m) / s
    w = np.linalg.solve(zs_.T @ zs_ + np.eye(16), zs_.T @ (v4 - v4.mean(0)))  # velocity readout in standardized z
    u = torch.from_numpy(w / s[:, None]).float().to(dev)  # back to raw z units
    u = u / u.norm(dim=0, keepdim=True)
    row["jac_vel_dir_gain"] = float((J @ u).square().sum((1, 2)).mean().sqrt() / np.sqrt(2))
    row["jac_mean_dir_gain"] = float((fro2 / 16).sqrt())
    rows.append(row)
dump(OUT / "g2.jsonl", rows)
print(f"g2 {arm} r{R} s{S} {time.time() - t0:.0f}s", flush=True)

# ---------- E: interventions at step 3000 ----------
_, model = load(arm, R, S, 3000, dev)
lo, hi = R - .5, 63.5 - R


def valid(c):
    return ((c >= lo) & (c <= hi)).all(axis=(1, 2))


def render(c):
    return disk_images(c, R, .3, 1.)


def frame_stats(out, c):
    """Per frame: brightness on the frame's own disk and on the other frame's disk (order index), centroids."""
    cov = disk_images(c, R, 0, 1) > .5
    o = out - .3
    own = np.stack([(o[:, k] * cov[:, k]).sum((1, 2)) / cov[:, k].sum((1, 2)) for k in (0, 1)], 1)
    other = np.stack([(o[:, k] * cov[:, 1 - k]).sum((1, 2)) / cov[:, 1 - k].sum((1, 2)) for k in (0, 1)], 1)
    return own, other


z9 = enc(model, x9)
base_out = dec(model, z9)
zall = enc(model, x4)
mu, icov = zall.mean(0), torch.linalg.pinv(torch.cov(zall.T))


def maha(z):
    d = z - mu
    return float(((d @ icov) * d).sum(1).sqrt().mean())


res = {"arm": arm, "radius": R, "seed": S, "base": image_metrics(base_out, c9, R), "maha_real": maha(z9)}
own, other = frame_stats(base_out, c9)
res["order_index"] = float(((own - other) / np.maximum(np.abs(own) + np.abs(other), 1e-6)).mean())
# swap
zsw = enc(model, x9[:, ::-1].copy())
out_sw = dec(model, zsw)
own_s, other_s = frame_stats(out_sw, c9)
res["swap"] = {"out_change_rms": float(np.sqrt(np.square(out_sw - base_out).mean())),
               "order_index_after": float(((own_s - other_s) / np.maximum(np.abs(own_s) + np.abs(other_s), 1e-6)).mean()),
               "z_change": float((zsw - z9).norm(dim=1).mean())}
# natural reference: decode a different real sample
perm = np.random.default_rng(5).permutation(1024)
res["natural_out_change_rms"] = float(np.sqrt(np.square(base_out[perm] - base_out).mean()))
# rotate displacement 90 degrees about the midpoint (in-distribution: same speed, components stay in range)
mid = c9.mean(1)
d = c9[:, 1] - c9[:, 0]
rot = np.stack((-d[:, 1], d[:, 0]), 1)
crot = np.stack((mid - rot / 2, mid + rot / 2), 1)
ok = valid(crot)
zr = enc(model, render(crot[ok]))
outr = dec(model, zr)
mr = image_metrics(outr, crot[ok], R)
res["rotate"] = {"n": int(ok.sum()), "z_change": float((zr - z9[ok]).norm(dim=1).mean()),
                 "out_change_rms": float(np.sqrt(np.square(outr - base_out[ok]).mean())), "metrics_on_target": mr}
# random direction with the same norm as the rotation edit
g = torch.Generator(device="cpu").manual_seed(7)
rd = torch.randn(int(ok.sum()), 16, generator=g).to(dev)
rd = rd / rd.norm(dim=1, keepdim=True) * (zr - z9[ok]).norm(dim=1, keepdim=True)
outrd = dec(model, z9[ok] + rd)
res["random_dir"] = {"out_change_rms": float(np.sqrt(np.square(outrd - base_out[ok]).mean())), "maha": maha(z9[ok] + rd)}
# latent interpolation halfway to the rotated pair
zi = (z9[ok] + zr) / 2
res["interp_half"] = {"maha": maha(zi), "out_change_rms": float(np.sqrt(np.square(dec(model, zi) - base_out[ok]).mean()))}
# position shift by +3 px in x (same displacement)
csh = c9 + np.array([3., 0.])
ok2 = valid(csh)
zsh = enc(model, render(csh[ok2]))
res["shift"] = {"n": int(ok2.sum()), "metrics_on_target": image_metrics(dec(model, zsh), csh[ok2], R),
                "out_change_rms": float(np.sqrt(np.square(dec(model, zsh) - base_out[ok2]).mean()))}
# amplify the antisymmetric component
zsym, zasym = (z9 + zsw) / 2, (z9 - zsw) / 2
amp = {}
for k in (1, 3, 10, 30, 100):
    ze = zsym + k * zasym
    o = dec(model, ze)
    a, b = frame_stats(o, c9)
    amp[str(k)] = {"order_index": float(((a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-6)).mean()),
                   "maha": maha(ze), "iou_ordered": image_metrics(o, c9, R)["iou"]}
res["asym_amplify"] = amp
dump(OUT / "interventions.jsonl", [res])
print(f"E {arm} r{R} s{S} {time.time() - t0:.0f}s", flush=True)
