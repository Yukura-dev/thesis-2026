"""Experiment B (endpoint study): which motion information a frozen encoder keeps. See PLAN.md."""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from data import MovieStream  # noqa: E402
from evaluation import ImageMetrics, predict_probes  # noqa: E402
from training import (Autoencoder, Config, _configure, _parameter_digest,  # noqa: E402
                      reconstruction_loss, resolve_device)

CACHE = ROOT / "checkpoints" / "b_endpoint"
LEDGER = OUT / "runs.jsonl"
GAP = 4
BASE = Config(frame_gap=GAP, steps=3000, lr=.003, reject_reflections=True)
CONDITIONS = {
    "r5_l2": (dict(radius=5, loss="l2"), "default"),
    "r5_l1": (dict(radius=5, loss="l1"), "default"),
    "r5_noisy_l1_s0.1": (dict(radius=5, loss="l1", noise_level=.1), "default"),
    "r5_noisy_l1_s0.4": (dict(radius=5, loss="l1", noise_level=.4), "default"),
    "r8_l1": (dict(radius=8, loss="l1"), "default"),
    "r8_l1_init_a": (dict(radius=8, loss="l1"), "a"),
    "r5_wl1_0.1": (dict(radius=5, loss="bg_weighted_l1", loss_scale=.1), "default"),
    "r5_wl1_0.01": (dict(radius=5, loss="bg_weighted_l1", loss_scale=.01), "default"),
    "r8_wl1_0.3": (dict(radius=8, loss="bg_weighted_l1", loss_scale=.3), "default"),
}
THRESHOLD_FRAME = .65  # thesis rule for a flat background


def ledger_shas():
    """SHA-256 of every matching historical model, keyed by (radius, loss, noise, scale, seed, init)."""
    found = {}
    for line in open(ROOT / "results" / "runs.jsonl"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = row.get("config", {})
        if not (c.get("frame_gap") == GAP and c.get("reject_reflections") and c.get("steps") == 3000
                and c.get("lr") == .003 and c.get("latent_dim", 16) == 16 and c.get("batch_size", 128) == 128
                and c.get("noise_type", "gaussian") == "gaussian" and c.get("placement", "target_only") == "target_only"
                and not c.get("background_texture_sd", 0) and "model_sha256" in row):
            continue
        runtime = row.get("runtime") if isinstance(row.get("runtime"), dict) else {}
        key = (c["radius"], c["loss"], c["noise_level"], c.get("loss_scale", 0.0), c["seed"],
               runtime.get("initialization", "default"))
        found.setdefault(key, set()).add(row["model_sha256"])
    for line in open(ROOT / "results" / "b3_weighted_l1" / "runs.jsonl"):
        row = json.loads(line)
        if row["w"] != 1.0:
            key = (row["radius"], "bg_weighted_l1", 0.0, row["w"], row["seed"], "default")
            found.setdefault(key, set()).add(row["model_sha256"])
    return found


def train(cfg, init, device):
    """Identical to training.run_experiment's loop; only the initialization may differ."""
    _configure(cfg.seed)
    model = Autoencoder(cfg.latent_dim, cfg.background, initialization=init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    model.train()
    for _ in range(cfg.steps):
        batch = stream.sample(cfg.batch_size)
        x = torch.from_numpy(batch.input).to(device)
        target = torch.from_numpy(batch.target).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = reconstruction_loss(model(x), target, cfg.loss, cfg.loss_scale)
        loss.backward()
        optimizer.step()
    return model.eval()


def collect(model, cfg, split, size, device, metrics=None):
    stream = MovieStream(np.random.SeedSequence([split, 0]), **cfg.movie_kwargs())
    zs, centers, labels = [], [], []
    with torch.inference_mode():
        for start in range(0, size, cfg.batch_size):
            batch = stream.sample(min(cfg.batch_size, size - start), need_target=False)
            z = model.encode(torch.from_numpy(batch.input).to(device))
            zs.append(z.cpu().numpy())
            if metrics is not None:
                metrics.update(model.decode(z).cpu().numpy(), batch, cfg.radius)
            centers.append(batch.centers.astype(np.float64))
            labels.append(batch.labels.astype(np.float64))
    return np.concatenate(zs), np.concatenate(centers), np.concatenate(labels)


def targets(centers, labels):
    prev, cur = centers[:, 0], centers[:, 1]
    d = cur - prev
    velocity = labels[:, 2:4]
    if not np.allclose(d / GAP, velocity, atol=1e-4):
        raise ValueError("Reflection-free windows must satisfy (cur - prev)/gap == velocity")
    sym = np.stack((d[:, 0]**2, d[:, 1]**2, d[:, 0] * d[:, 1]), axis=1)
    return {"ends": np.concatenate((prev, cur), axis=1), "velocity": velocity,
            "mid": (prev + cur) / 2, "sym": sym, "d": d}


def r2(truth, pred):
    truth, pred = np.asarray(truth, np.float64), np.asarray(pred, np.float64)
    return float(np.mean(1 - np.square(truth - pred).mean(0) / truth.var(0)))


def fit_big(x, y, xt):
    """Larger fixed MLP probe (256-256, 200 epochs) on standardized inputs and targets."""
    xm, xs = x.mean(0), x.std(0)
    xs[xs < 1e-12] = 1
    ym, ys = y.mean(0), y.std(0)
    f = torch.from_numpy(((x - xm) / xs).astype(np.float32))
    t = torch.from_numpy(((y - ym) / ys).astype(np.float32))
    ft = torch.from_numpy(((xt - xm) / xs).astype(np.float32))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(207)
            shuffle = torch.Generator().manual_seed(208)
            model = nn.Sequential(nn.Linear(f.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, t.shape[1]))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
            for _ in range(200):
                for ids in torch.randperm(len(f), generator=shuffle).split(256):
                    opt.zero_grad(set_to_none=True)
                    nn.functional.mse_loss(model(f[ids]), t[ids]).backward()
                    opt.step()
                sched.step()
            with torch.inference_mode():
                return model(ft).numpy() * ys + ym
    finally:
        torch.set_num_threads(threads)


def slope(err, d):
    """Mean over axes of the least-squares slope of endpoint error on true displacement."""
    dc = d - d.mean(0)
    ec = err - err.mean(0)
    return float(np.mean((ec * dc).sum(0) / (dc * dc).sum(0)))


def analyse(z, tr, zt, te, big=True):
    out = {}
    for method in ("ridge", "mlp"):
        out[method] = {}
    preds = {name: predict_probes(z, tr[name], zt) for name in ("ends", "velocity", "mid", "sym")}
    for method in ("ridge", "mlp"):
        ends = preds["ends"][method]
        p_prev, p_cur = ends[:, :2], ends[:, 2:]
        t_prev, t_cur = te["ends"][:, :2], te["ends"][:, 2:]
        e_prev, e_cur = p_prev - t_prev, p_cur - t_cur
        out[method] = {
            "prev": r2(t_prev, p_prev), "cur": r2(t_cur, p_cur),
            "v_delta": r2(te["velocity"], (p_cur - p_prev) / GAP),
            "v_direct": r2(te["velocity"], preds["velocity"][method]),
            "mid": r2(te["mid"], preds["mid"][method]),
            "sym": r2(te["sym"], preds["sym"][method]),
            "rmse_prev": float(np.sqrt(np.square(e_prev).mean())),
            "rmse_cur": float(np.sqrt(np.square(e_cur).mean())),
            "slope_prev": slope(e_prev, te["d"]), "slope_cur": slope(e_cur, te["d"]),
            "error_corr": float(np.mean([np.corrcoef(e_prev[:, k], e_cur[:, k])[0, 1] for k in (0, 1)])),
        }
    if big:
        out["big"] = {"v_direct": r2(te["velocity"], fit_big(z, tr["velocity"], zt)),
                      "sym": r2(te["sym"], fit_big(z, tr["sym"], zt))}
    return out


def classify(m):
    if m["v_direct"] >= .8:
        return "ordered"
    if m["v_direct"] < .5 and m["sym"] >= .5 and m["mid"] >= .9:
        return "unordered"
    if m["v_direct"] < .5 and m["sym"] < .5 and max(m["prev"], m["cur"]) >= .9:
        return "one_endpoint"
    return "partial"


def references(radius):
    """Same probes on oracle inputs: the true current position, and the true unordered pair."""
    cfg = replace(BASE, radius=radius)
    data = {}
    for split, size in ((2, 10000), (3, 5000)):
        stream = MovieStream(np.random.SeedSequence([split, 0]), **cfg.movie_kwargs())
        c, lab = stream.geometry(size)
        data[split] = targets(np.asarray(c, np.float64), np.asarray(lab, np.float64))
    result = {}
    for name, feature in (("current_only", lambda t: t["ends"][:, 2:]),
                          ("unordered_pair", lambda t: np.concatenate((t["mid"], t["sym"]), 1))):
        result[name] = analyse(feature(data[2]), data[2], feature(data[3]), data[3], big=False)["mlp"]
    return result


def done_keys():
    if not LEDGER.exists():
        return set()
    return {(r["condition"], r["seed"]) for r in map(json.loads, LEDGER.read_text().splitlines()) if r}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    device = resolve_device("cuda")
    ref_path = OUT / "references.json"
    if not ref_path.exists():
        ref_path.write_text(json.dumps({str(r): references(r) for r in (5, 8)}, indent=1) + "\n")
        print("references written", flush=True)
    shas = ledger_shas()
    done = done_keys()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in args.conditions:
        overrides, init = CONDITIONS[name]
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            start = time.time()
            cfg = replace(BASE, seed=seed, **overrides)
            model = train(cfg, init, device)
            sha = _parameter_digest(model)
            expected = shas.get((cfg.radius, cfg.loss, cfg.noise_level, cfg.loss_scale, seed, init))
            metrics = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
            z, c, lab = collect(model, cfg, 2, cfg.probe_train, device)
            zt, ct, labt = collect(model, cfg, 3, cfg.probe_test, device, metrics)
            image = metrics.result()
            frames = ("background", "current_only", "past_only", "both")[
                2 * (image["center_intensity_previous"] >= THRESHOLD_FRAME)
                + (image["center_intensity_current"] >= THRESHOLD_FRAME)]
            probes = analyse(z, targets(c, lab), zt, targets(ct, labt))
            torch.save(model.state_dict(), CACHE / f"{name}_s{seed}.pt")
            np.savez_compressed(CACHE / f"{name}_s{seed}_latents.npz", z=z, zt=zt, c=c, ct=ct, lab=lab, labt=labt)
            row = {"condition": name, "seed": seed, "initialization": init, "config": cfg.__dict__,
                   "model_sha256": sha, "ledger_match": None if expected is None else sha in expected,
                   "frames": frames, "image": image, "probes": probes,
                   "class": classify(probes["mlp"]), "seconds": time.time() - start}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(row) + "\n")
            m = probes["mlp"]
            print(name, seed, row["ledger_match"], frames, row["class"],
                  *(f"{k}={m[k]:.3f}" for k in ("prev", "cur", "v_direct", "v_delta", "mid", "sym",
                                                 "slope_prev", "slope_cur")),
                  f"big_v={probes['big']['v_direct']:.3f}", f"big_sym={probes['big']['sym']:.3f}",
                  f"{row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
