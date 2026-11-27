"""G3: frozen encoders, fresh decoders retrained under many latent maps (PLAN.md). One (radius, seed) per call.

All decoders of one call see the same training batches (the original run's stream) and start from the same
seed-matched initialization; they differ only in encoder, latent map, loss, optimizer or learning rate.
For speed the decoders are stacked and trained together with torch.func.vmap; Adam (torch defaults: betas
0.9/0.999, eps 1e-8, bias correction) and SGD with momentum 0.9 are written out elementwise so each decoder can
have its own optimizer and learning rate. The summed loss has block-diagonal gradients, so the decoders are
independent.
"""
import argparse
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state

from mech import OUT, Autoencoder, MovieStream, _configure, arm_config, dump, image_metrics, load, sample

p = argparse.ArgumentParser()
p.add_argument("radius", type=int)
p.add_argument("seed", type=int)
p.add_argument("--steps", type=int, default=3000)
a = p.parse_args()
dev = torch.device("cuda")
R, S = a.radius, a.seed
ENCODERS = {"untrained": ("l1", 0), "l1": ("l1", 3000), "wl1": ("wl1", 3000), "gauss": ("gauss", 3000),
            "l2": ("l2", 3000)}
ENC_NAMES = list(ENCODERS)
PRE = ("none", "center", "scale", "standardize", "whiten", "randlin")
CONTROL_ENC = ("untrained", "l1", "l2")
LOG_STEPS = (1, 10, 100, 1000, 3000)


class Decoder(nn.Module):
    def __init__(self, ae, A, c):
        super().__init__()
        self.from_latent = copy.deepcopy(ae.from_latent)
        self.decoder = copy.deepcopy(ae.decoder)
        self.bg = ae.background
        self.register_buffer("A", A.clone())
        self.register_buffer("c", c.clone())

    def forward(self, z):
        return self.bg + self.decoder(self.from_latent(z @ self.A.T + self.c).reshape(-1, 32, 4, 4))

    @torch.no_grad()
    def reparam(self, A_new, c_new):
        """Change the latent map but keep the decoder's function: W A z + W c + b == W' A' z + W' c' + b'."""
        W, b = self.from_latent.weight, self.from_latent.bias
        W_new = W @ self.A @ torch.linalg.inv(A_new)
        b_new = b + W @ self.c - W_new @ c_new
        W.copy_(W_new)
        b.copy_(b_new)
        self.A.copy_(A_new)
        self.c.copy_(c_new)


def maps(enc_model, cfg):
    """Each latent map as (A, c) with T(z) = A z + c; statistics from split 4 only."""
    x, _, _ = sample(cfg, 4, 8192)
    with torch.no_grad():
        z = torch.cat([enc_model.encode(torch.from_numpy(x[i:i + 1024]).to(dev)) for i in range(0, 8192, 1024)]).double()
    m, sd = z.mean(0), z.std(0)
    tiny = sd < 1e-6 * sd.median()
    sd = torch.where(tiny, torch.ones_like(sd), sd)
    eye = torch.eye(16, dtype=torch.float64, device=dev)
    lam, U = torch.linalg.eigh(torch.cov(z.T))
    lam = lam.clamp_min(1e-6 * lam.max())
    zca = U @ torch.diag(lam.rsqrt()) @ U.T
    g = torch.Generator().manual_seed(1000 * R + S)
    G = torch.randn(16, 16, generator=g, dtype=torch.float64).to(dev)
    G = G / G.norm(dim=0, keepdim=True)
    out = {"none": (eye, 0 * m), "center": (eye, -m), "scale": (torch.diag(1 / sd), 0 * m),
           "standardize": (torch.diag(1 / sd), -m / sd), "whiten": (zca, -zca @ m),
           "randlin": (G @ torch.diag(1 / sd), 0 * m)}
    info = {"tiny_dims": int(tiny.sum()), "sd_min": float(sd.min()), "sd_max": float(sd.max()),
            "mean_norm": float(m.norm()), "sd_rms": float(sd.square().mean().sqrt()),
            "cov_cond": float(lam.max() / lam.min()), "randlin_cond": float(torch.linalg.cond(G))}
    return {k: (A.float(), c.float()) for k, (A, c) in out.items()}, info


cfg = arm_config("l1", R, S)
encoders, map_info = {}, {}
for name, (arm, step) in ENCODERS.items():
    _, model = load(arm, R, S, step, dev)
    encoders[name] = model
    map_info[name] = maps(model, cfg)
_configure(S)
import os  # noqa: E402
if os.environ.get("G3_FAST", "1") == "1":  # speed: cuDNN autotuning, nondeterministic sums; TF32 stays off
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
init = Autoencoder(cfg.latent_dim, cfg.background).to(dev)  # identical to the runs' initial decoder

keys, decs = [], []


def add(enc, pre, loss, opt="adam", lr=3e-3, absorbed=None):
    A, c = map_info[enc][0][pre]
    if absorbed:  # start from the other map's function, then switch parametrization
        A0, c0 = map_info[enc][0][absorbed]
        d = Decoder(init, A0, c0).to(dev)
        d.reparam(A, c)
    else:
        d = Decoder(init, A, c).to(dev)
    keys.append({"encoder": enc, "pre": pre, "dec_loss": loss, "opt": opt, "lr": lr, "absorbed_from": absorbed})
    decs.append(d)


for enc in ENCODERS:
    for pre in PRE:
        for loss in ("l1", "l2"):
            add(enc, pre, loss)
for enc in CONTROL_ENC:
    for loss in ("l1", "l2"):
        add(enc, "none", loss, absorbed="standardize")
        add(enc, "standardize", loss, absorbed="none")
        add(enc, "none", loss, lr=3e-4)
        add(enc, "none", loss, lr=3e-2)
        for pre in ("none", "standardize"):
            for lr in (1e-2, 1e-1):
                add(enc, pre, loss, opt="sgd", lr=lr)

N = len(decs)
params, buffers = stack_module_state(decs)
params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
base = copy.deepcopy(decs[0]).to("meta")


def f(pp, bb, z):
    return functional_call(base, (pp, bb), (z,))


F_ = torch.nn.functional


def fwd(pp, bb, Z):
    """All decoders at once with grouped transposed convolutions. Z: (N, B, 16) -> (N, B, 2, 64, 64)."""
    n, b = Z.shape[:2]
    zt = torch.einsum("nij,nbj->nbi", bb["A"], Z) + bb["c"][:, None]
    h = torch.einsum("noi,nbi->nbo", pp["from_latent.weight"], zt) + pp["from_latent.bias"][:, None]
    x = h.permute(1, 0, 2).reshape(b, n * 32, 4, 4)
    for conv, gn in ((0, 1), (3, 4), (6, 7), (9, None)):
        w = pp[f"decoder.{conv}.weight"]
        bias = pp[f"decoder.{conv}.bias"].reshape(-1) if f"decoder.{conv}.bias" in pp else None
        x = F_.conv_transpose2d(x, w.reshape(-1, *w.shape[2:]), bias, stride=2, padding=1, groups=n)
        if gn is not None:
            C, H, W = w.shape[2], x.shape[2], x.shape[3]
            y = x.view(b, n, C * H * W)
            mu = y.mean(2, keepdim=True)
            var = y.var(2, unbiased=False, keepdim=True)
            y = ((y - mu) / (var + 1e-5).sqrt()).view(b, n, C, H, W)
            y = y * pp[f"decoder.{gn}.weight"][None, :, :, None, None] + pp[f"decoder.{gn}.bias"][None, :, :, None, None]
            x = F_.leaky_relu(y, .1).reshape(b, n * C, H, W)
    return .3 + x.view(b, n, 2, 64, 64).transpose(0, 1)


with torch.no_grad():  # the grouped forward must equal each decoder's own forward
    zt = torch.randn(N, 8, 16, device=dev)
    ref = torch.stack([decs[i](zt[i]) for i in range(N)])
    gap = float((fwd(params, buffers, zt) - ref).abs().max())
    assert gap < 1e-4, gap
    print("grouped-vs-module max abs diff", gap, flush=True)
enc_idx = torch.tensor([ENC_NAMES.index(k["encoder"]) for k in keys], device=dev)
is_l1 = torch.tensor([k["dec_loss"] == "l1" for k in keys], device=dev)
is_adam = torch.tensor([k["opt"] == "adam" for k in keys], device=dev)
lr = torch.tensor([k["lr"] for k in keys], device=dev)
state = {k: (torch.zeros_like(v), torch.zeros_like(v)) for k, v in params.items()}  # Adam m, v or SGD momentum


def bcast(t, ref):
    return t.view(-1, *([1] * (ref.dim() - 1)))


data = MovieStream(np.random.SeedSequence([0, S]), **cfg.movie_kwargs())
curves = [[] for _ in range(N)]
diag = [[] for _ in range(N)]
t0 = time.time()
for step in range(1, a.steps + 1):
    batch = data.sample(cfg.batch_size)
    x = torch.from_numpy(batch.input).to(dev)
    target = torch.from_numpy(batch.target).to(dev)
    with torch.no_grad():
        Z = torch.stack([encoders[n].encode(x) for n in ENC_NAMES])[enc_idx]  # (N, B, 16)
    out = fwd(params, buffers, Z)
    err = out - target
    per = torch.where(is_l1, err.abs().mean((1, 2, 3, 4)), err.square().mean((1, 2, 3, 4)))
    grads = torch.autograd.grad(per.sum(), list(params.values()))
    watch = step in LOG_STEPS
    with torch.no_grad():
        upd_sq = {}
        b1, b2, eps = .9, .999, 1e-8
        for (name, p_), g in zip(params.items(), grads):
            m, v = state[name]
            # Adam
            m_a = b1 * m + (1 - b1) * g
            v_a = b2 * v + (1 - b2) * g * g
            step_a = (m_a / (1 - b1 ** step)) / ((v_a / (1 - b2 ** step)).sqrt() + eps)
            # SGD with momentum (torch: buf = g at the first step, then mu*buf + g)
            buf = g.clone() if step == 1 else .9 * m + g
            ad = bcast(is_adam, p_)
            new_m = torch.where(ad, m_a, buf)
            new_v = torch.where(ad, v_a, v)
            delta = -bcast(lr, p_) * torch.where(ad, step_a, buf)
            p_.add_(delta)
            state[name] = (new_m, new_v)
            if watch:
                upd_sq[name] = (delta.flatten(1).norm(dim=1), g.flatten(1).norm(dim=1))
        if watch:
            for i in range(N):
                diag[i].append({"step": step, "grad": {n: float(u[1][i]) for n, u in upd_sq.items()},
                                "update": {n: float(u[0][i]) for n, u in upd_sq.items()}})
        if step % 25 == 0 or step == 1:
            pl = per.tolist()
            for i in range(N):
                curves[i].append((step, pl[i]))
    if step % 500 == 0 or step == 50:
        print(f"r{R} s{S} step {step} {time.time() - t0:.0f}s", flush=True)

xe, ce, _ = sample(cfg, 9, 1024)
with torch.no_grad():
    ze = {n: torch.cat([encoders[n].encode(torch.from_numpy(xe[j:j + 256]).to(dev)) for j in range(0, 1024, 256)])
          for n in ENC_NAMES}
rows = []
for i, key in enumerate(keys):
    pi = {k: v[i] for k, v in params.items()}
    bi = {k: v[i] for k, v in buffers.items()}
    with torch.no_grad():
        out = torch.cat([f(pi, bi, ze[key["encoder"]][j:j + 256]) for j in range(0, 1024, 256)]).cpu().numpy()
    rows.append({"radius": R, "seed": S, **key, "metrics": image_metrics(out, ce, R), "curve": curves[i],
                 "diag": diag[i], "finite": bool(np.isfinite(out).all())})
for enc, (_, info) in map_info.items():
    rows.append({"radius": R, "seed": S, "map_info": enc, **info})
dump(OUT / ("g3.jsonl" if a.steps == 3000 else "g3_test.jsonl"), rows)
print("done", R, S, time.time() - t0, flush=True)
