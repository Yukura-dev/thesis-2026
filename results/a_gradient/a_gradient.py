"""Experiment A (gradient study): background vs object gradients along real Adam trajectories. See PLAN.md."""
import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from data import MovieStream  # noqa: E402
from training import (Autoencoder, Config, _configure, _parameter_digest,  # noqa: E402
                      gaussian_smoothed_l1, reconstruction_loss, resolve_device)

LEDGER = OUT / "runs.jsonl"
BASE = Config(radius=5, frame_gap=4, steps=3000, lr=.003, reject_reflections=True)
CHECKPOINTS = (0, 50, 300, 1000, 2980)
WINDOW = 20
TRACE_EVERY = 25
EVAL_SPLIT, EVAL_SIZE = 9, 512


def elementwise(prediction, target, loss, scale):
    error = prediction - target
    if loss == "l1":
        return error.abs()
    if loss == "gauss_l1":
        return gaussian_smoothed_l1(error, scale)
    if loss == "l2":
        return error.square()
    raise ValueError(loss)


def masks(target):
    """Background: clean pixels equal to the background value. Core: fully covered. Edge: partial coverage."""
    low = target.amin(dim=(-2, -1), keepdim=True)
    high = target.amax(dim=(-2, -1), keepdim=True)
    background = target <= low
    core = target >= high
    return {"B": background, "E": ~background & ~core, "K": core}


def parts(model, x, target, loss, scale):
    """Losses per pixel group, each divided by the total pixel count so the parts sum to the mean loss."""
    values = elementwise(model(x), target, loss, scale)
    n = values.numel()
    out ={k: (values * m).sum() / n for k, m in masks(target).items()}
    total = values.mean()
    return out, total


def flat_grads(model, x, target, loss, scale):
    params = [p for p in model.parameters()]
    losses, total = parts(model, x, target, loss, scale)
    grads = {}
    for k, value in losses.items():
        g = torch.autograd.grad(value, params, retain_graph=True)
        grads[k] = torch.cat([t.reshape(-1) for t in g]).double()
    grads["C"] = grads["E"] + grads["K"]
    g_total = torch.cat([t.reshape(-1) for t in torch.autograd.grad(total, params)]).double()
    if (grads["B"] + grads["C"] - g_total).norm() > 1e-5 * g_total.norm() + 1e-12:
        raise ArithmeticError("Group gradients do not add up to the total gradient")
    return {k: float(v.item()) for k, v in losses.items()}, grads, g_total


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-300))


def snapshot(model, x, target, loss, scale, other):
    """Gradient geometry at the current weights, under the arm's loss and under a counterfactual loss."""
    result = {}
    for label, (name, s) in (("own", (loss, scale)), ("counterfactual", other)):
        values, g, total = flat_grads(model, x, target, name, s)
        # Directional derivatives along plain steepest descent of this loss, per unit step length.
        direction = -total / (total.norm() + 1e-300)
        result[label] = {"loss": name, "scale": s, "parts": values,
                         "norm": {k: float(g[k].norm()) for k in ("B", "C", "E", "K")},
                         "cos_BC": cos(g["B"], g["C"]),
                         "sd_dB": float(g["B"] @ direction), "sd_dC": float(g["C"] @ direction)}
    return result


def image_stats(model, x, target):
    with torch.no_grad():
        out = model(x)
    m = masks(target)
    return {"core_mean": float(out[m["K"]].mean()), "background_abs_error": float((out - target).abs()[m["B"]].mean()),
            "core_abs_error": float((out - target).abs()[m["K"]].mean())}


def run(loss, scale, seed, device, other):
    cfg = replace(BASE, loss=loss, loss_scale=scale, seed=seed)
    eval_batch = MovieStream(np.random.SeedSequence([EVAL_SPLIT, 0]), **cfg.movie_kwargs()).sample(EVAL_SIZE, need_target=False)
    ex = torch.from_numpy(eval_batch.input).to(device)
    et = torch.from_numpy(eval_batch.clean).to(device)
    _configure(cfg.seed)
    model = Autoencoder(cfg.latent_dim, cfg.background).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    model.train()
    trace, checkpoints, window = [], {}, None
    for step in range(cfg.steps):
        if step % TRACE_EVERY == 0:
            with torch.no_grad():
                values, total = parts(model, ex, et, loss, scale)
            trace.append({"step": step, **{k: float(v) for k, v in values.items()}, **image_stats(model, ex, et)})
        if step in CHECKPOINTS:
            checkpoints[step] = {"snapshot": snapshot(model, ex, et, loss, scale, other),
                                 "image": image_stats(model, ex, et)}
            window = {"start": step, "dB": 0.0, "dE": 0.0, "dK": 0.0, "update_norm": 0.0,
                      "cos_u_gB": [], "cos_u_gC": [],
                      "parts_before": checkpoints[step]["snapshot"]["own"]["parts"]}
        if window is not None:
            _, g, _ = flat_grads(model, ex, et, loss, scale)
            before = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double()
        batch = stream.sample(cfg.batch_size)
        x = torch.from_numpy(batch.input).to(device)
        target = torch.from_numpy(batch.target).to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction_loss(model(x), target, cfg.loss, cfg.loss_scale).backward()
        optimizer.step()
        if window is not None:
            u = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double() - before
            for k in ("B", "E", "K"):
                window["d" + k] += float(g[k] @ u)
            window["update_norm"] += float(u.norm())
            window["cos_u_gB"].append(cos(-u, g["B"]))
            window["cos_u_gC"].append(cos(-u, g["C"]))
            if step + 1 - window["start"] == WINDOW:
                with torch.no_grad():
                    after, _ = parts(model, ex, et, loss, scale)
                w = checkpoints[window.pop("start")]
                window["parts_after"] = {k: float(v) for k, v in after.items()}
                window["cos_u_gB"] = float(np.mean(window["cos_u_gB"]))
                window["cos_u_gC"] = float(np.mean(window["cos_u_gC"]))
                w["adam_window"] = window
                window = None
    return cfg, model, trace, checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["l1:0", "gauss_l1:0.05"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--counterfactual", default="gauss_l1:0.05")
    args = parser.parse_args()
    device = resolve_device("cuda")
    done = set()
    if LEDGER.exists():
        done = {(r["loss"], r["scale"], r["seed"]) for r in map(json.loads, LEDGER.read_text().splitlines())}
    for arm in args.arms:
        loss, scale = arm.split(":")
        scale = float(scale)
        cf_loss, cf_scale = args.counterfactual.split(":")
        other = (cf_loss, float(cf_scale)) if cf_loss != loss else ("l1", 0.0)
        for seed in args.seeds:
            if (loss, scale, seed) in done:
                continue
            start = time.time()
            cfg, model, trace, checkpoints = run(loss, scale, seed, device, other)
            row = {"loss": loss, "scale": scale, "seed": seed, "counterfactual": list(other),
                   "model_sha256": _parameter_digest(model), "checkpoints": checkpoints, "trace": trace,
                   "seconds": time.time() - start}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(row) + "\n")
            last = checkpoints[2980]["adam_window"]
            print(loss, scale, seed, row["model_sha256"][:12], f"dB={last['dB']:.2e} dE={last['dE']:.2e} dK={last['dK']:.2e}",
                  f"core={checkpoints[2980]['image']['core_mean']:.3f}", f"{row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
