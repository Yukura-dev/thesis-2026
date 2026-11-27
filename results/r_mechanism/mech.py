"""Shared code for Experiment R (see PLAN.md): arms, training with checkpoints, data, metrics, probes."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results" / "b_endpoint"))
from data import MovieStream, SIZE, disk_images  # noqa: E402
from evaluation import _fit_mlp  # noqa: E402
from training import Autoencoder, Config, _configure, _parameter_digest, reconstruction_loss  # noqa: E402
from b_endpoint import ledger_shas, fit_big  # noqa: E402

CKPT = ROOT / "checkpoints" / "r_mechanism"
GAP = 4
BASE = Config(frame_gap=GAP, steps=3000, lr=.003, reject_reflections=True)
STEPS = (0, 1, 2, 5, 10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000)
RADII = (5, 8)
SEEDS = (0, 1, 2, 3, 4)
WL1_W = {5: .1, 8: .3}


def arm_config(arm, radius, seed):
    """Thesis Config for each arm. G4 arms and wl1_ord train plain/weighted L1 with an extra element."""
    base = replace(BASE, radius=radius, seed=seed)
    loss = {"l1": dict(loss="l1"), "l2": dict(loss="l2"), "gauss": dict(loss="gauss_l1", loss_scale=.4),
            "wl1": dict(loss="bg_weighted_l1", loss_scale=WL1_W[radius]),
            "wl1_ord": dict(loss="bg_weighted_l1", loss_scale=WL1_W[radius]),
            "l1_affine": dict(loss="l1"), "l1_bn": dict(loss="l1"), "l1_switch300": dict(loss="l1")}[arm]
    return replace(base, **loss)


BASE_ARMS = ("l1", "wl1", "gauss", "l2")
EXTRA_ARMS = ("wl1_ord", "l1_affine", "l1_bn", "l1_switch300")


def run_dir(arm, radius, seed):
    return CKPT / f"{arm}_r{radius}_s{seed}"


def stream(cfg, split):
    return MovieStream(np.random.SeedSequence([split, 0]), **cfg.movie_kwargs())


def sample(cfg, split, n):
    """Deterministic fixed set: inputs, clean images, centres (n,2,2) and velocity labels."""
    b = stream(cfg, split).sample(n, need_target=False)
    return b.input, b.centers.astype(np.float64), b.labels[:, 2:4].astype(np.float64)


def wl1_ord_loss(pred, target, w):
    """bg_weighted_l1 except that pixels of the OTHER frame's disk keep weight 1 (PLAN: W-union test)."""
    floor = target.amin(dim=(-2, -1), keepdim=True)
    background = target <= floor
    other_disk = (target > floor).flip(1)
    weight = torch.where(background & ~other_disk, torch.full_like(target, w), torch.ones_like(target))
    return (weight * (pred - target).abs()).mean()


class Wrapped(nn.Module):
    """Autoencoder with an optional fixed affine map or BatchNorm between encoder and decoder."""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae
        self.register_buffer("shift", torch.zeros(ae.to_latent.out_features))
        self.register_buffer("scale", torch.ones(ae.to_latent.out_features))
        self.bn = None

    def encode(self, x):
        return self.ae.encode(x)

    def decode(self, z):
        z = (z - self.shift) / self.scale
        if self.bn is not None:
            z = self.bn(z)
        return self.ae.decode(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    @torch.no_grad()
    def absorb_standardization(self, mean, sd):
        """Insert z' = (z - mean)/sd and rescale from_latent so the decoder's function is unchanged."""
        layer = self.ae.from_latent
        layer.bias.add_(layer.weight @ mean)
        layer.weight.mul_(sd[None, :])
        self.shift.copy_(mean)
        self.scale.copy_(sd)


@torch.no_grad()
def latent_stats(model, cfg, device, n=8192, split=4):
    x, _, _ = sample(cfg, split, n)
    z = torch.cat([model.encode(torch.from_numpy(x[i:i + 1024]).to(device)) for i in range(0, n, 1024)])
    sd = z.std(0)
    sd = torch.where(sd < 1e-6 * sd.median(), torch.ones_like(sd), sd)
    return z.mean(0), sd


def group_norms(model):
    enc, dec = [], []
    for name, p in model.named_parameters():
        (enc if ("encoder" in name or "to_latent" in name) else dec).append(p)
    return enc, dec


def train_run(arm, radius, seed, device, log_every=1):
    """Thesis loop (identical to b_endpoint.train for base arms) with checkpoints and logging."""
    cfg = arm_config(arm, radius, seed)
    _configure(cfg.seed)
    ae = Autoencoder(cfg.latent_dim, cfg.background).to(device)
    model = Wrapped(ae).to(device)
    if arm == "l1_affine":
        model.eval()
        model.absorb_standardization(*latent_stats(model, cfg, device))
    if arm == "l1_bn":
        model.bn = nn.BatchNorm1d(cfg.latent_dim, affine=False).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.lr, foreach=False)
    data = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    out = run_dir(arm, radius, seed)
    out.mkdir(parents=True, exist_ok=True)
    enc, dec = group_norms(ae)
    log = []
    torch.save(model.state_dict(), out / "0.pt")
    for step in range(1, cfg.steps + 1):
        if arm == "l1_switch300" and step == 301:
            model.eval()
            model.absorb_standardization(*latent_stats(model, cfg, device))
            optimizer.state.pop(ae.from_latent.weight, None)
            optimizer.state.pop(ae.from_latent.bias, None)
        model.train()
        batch = data.sample(cfg.batch_size)
        x = torch.from_numpy(batch.input).to(device)
        target = torch.from_numpy(batch.target).to(device)
        before = [p.detach().clone() for p in enc + dec]
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        if arm == "wl1_ord":
            loss = wl1_ord_loss(pred, target, cfg.loss_scale)
        else:
            loss = reconstruction_loss(pred, target, cfg.loss, cfg.loss_scale)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            swapped = reconstruction_loss(pred, target.flip(1), cfg.loss, cfg.loss_scale)
            du = [(p - q).square().sum() for p, q in zip(enc + dec, before)]
            ne = len(enc)
            log.append((step, loss.item(), swapped.item(), float(sum(du[:ne]).sqrt()), float(sum(du[ne:]).sqrt())))
        if step in STEPS:
            torch.save(model.state_dict(), out / f"{step}.pt")
    np.save(out / "steplog.npy", np.array(log))
    return cfg, model, _parameter_digest(ae)


def load(arm, radius, seed, step, device):
    cfg = arm_config(arm, radius, seed)
    model = Wrapped(Autoencoder(cfg.latent_dim, cfg.background)).to(device)
    state = torch.load(run_dir(arm, radius, seed) / f"{step}.pt", map_location=device)
    if any(k.startswith("bn.") for k in state):
        model.bn = nn.BatchNorm1d(cfg.latent_dim, affine=False).to(device)
    model.load_state_dict(state)
    return cfg, model.eval()


# ---------- image metrics ----------
def image_metrics(out, centers, radius):
    """out (n,2,64,64) numpy; centers (n,2,2). Thresholds from PLAN.md."""
    cov = disk_images(centers, radius, 0, 1)
    truth = cov > .5
    pred = out > .65
    inter = (truth & pred).sum((2, 3))
    union = (truth | pred).sum((2, 3))
    iou = inter / np.maximum(union, 1)
    dice = 2 * inter / np.maximum(truth.sum((2, 3)) + pred.sum((2, 3)), 1)
    fg = cov > 0
    err = np.abs(out - (.3 + .7 * cov))
    fg_mae = (err * fg).sum((2, 3)) / fg.sum((2, 3))
    bg_mae = (err * ~fg).sum((2, 3)) / (~fg).sum((2, 3))
    mass = np.clip(out - .3, 0, None) * pred
    m = mass.sum((2, 3))
    detected = m > 0
    g = np.arange(SIZE)
    cx = (mass * g[None, None, None, :]).sum((2, 3)) / np.maximum(m, 1e-12)
    cy = (mass * g[None, None, :, None]).sum((2, 3)) / np.maximum(m, 1e-12)
    cen = np.stack((cx, cy), -1)
    cerr = np.linalg.norm(cen - centers, axis=-1)
    both = detected.all(1)
    d_true = centers[:, 1] - centers[:, 0]
    d_err = np.linalg.norm((cen[:, 1] - cen[:, 0]) - d_true, axis=-1)
    target_sd = (.3 + .7 * cov).std((2, 3))
    return {
        "iou": float(iou.mean()), "iou_prev": float(iou[:, 0].mean()), "iou_cur": float(iou[:, 1].mean()),
        "dice": float(dice.mean()), "fg_mae": float(fg_mae.mean()), "bg_mae": float(bg_mae.mean()),
        "detect_fail": float(1 - detected.mean()),
        "center_err": float(cerr[detected].mean()) if detected.any() else None,
        "disp_err": float(d_err[both].mean()) if both.any() else None,
        "out_sd": float(out.std((2, 3)).mean()), "target_sd": float(target_sd.mean()),
        "center_int_prev": float(bilinear(out[:, 0], centers[:, 0]).mean()),
        "center_int_cur": float(bilinear(out[:, 1], centers[:, 1]).mean()),
    }


def bilinear(img, c):
    lo = np.floor(c).astype(int)
    dx, dy = (c - lo).T
    x, y = lo.T
    k = np.arange(len(img))
    return ((1 - dy) * ((1 - dx) * img[k, y, x] + dx * img[k, y, x + 1]) +
            dy * ((1 - dx) * img[k, y + 1, x] + dx * img[k, y + 1, x + 1]))


def collapsed(m):
    return m["center_int_prev"] < .65 and m["center_int_cur"] < .65 and m["out_sd"] < .1 * m["target_sd"]


# ---------- probes ----------
def targets(centers, v):
    prev, cur = centers[:, 0], centers[:, 1]
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    return {"prev": prev, "cur": cur, "mid": (prev + cur) / 2, "vel": v,
            "dir": v / np.maximum(sp, 1e-9)}


def r2(t, p):
    t, p = np.asarray(t, np.float64), np.asarray(p, np.float64)
    return float(np.mean(1 - np.square(t - p).mean(0) / t.var(0)))


def standardize(x, xt):
    m, s = x.mean(0), x.std(0)
    s[s < 1e-12] = 1
    return (x - m) / s, (xt - m) / s


def ridge(x, y, xt, alpha=1.0):
    x, xt = standardize(np.asarray(x, np.float64), np.asarray(xt, np.float64))
    ym = y.mean(0)
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - ym))
    return xt @ w + ym


def ridge_cv(x, y, xt, alphas=(1e-2, 1, 1e2, 1e4)):
    """Alpha chosen on the last 20 % of the training set; refit on all of it."""
    n = int(.8 * len(x))
    best = min(alphas, key=lambda a: np.square(ridge(x[:n], y[:n], x[n:], a) - y[n:]).mean() / y[n:].var(0).mean())
    return ridge(x, y, xt, best), best


def mlp(x, y, xt):
    """Thesis probe (64-64, 80 epochs) on standardized inputs and targets."""
    x, xt = standardize(np.asarray(x, np.float64), np.asarray(xt, np.float64))
    ym, ys = y.mean(0), y.std(0)
    return _fit_mlp(x, (y - ym) / ys, xt) * ys + ym


def probe_all(ztr, ttr, zte, tte, kinds=("ridge", "mlp"), keys=("prev", "cur", "mid", "vel", "dir")):
    """One joint fit per probe kind over all targets, scored per target."""
    y = np.concatenate([ttr[k] for k in keys], 1)
    res = {}
    for kind in kinds:
        p = {"ridge": lambda: ridge(ztr, y, zte), "mlp": lambda: mlp(ztr, y, zte),
             "big": lambda: fit_big(ztr, y, zte)}[kind]()
        for i, k in enumerate(keys):
            res[f"{k}_{kind}"] = r2(tte[k], p[:, 2 * i:2 * i + 2])
    return res


def dump(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
