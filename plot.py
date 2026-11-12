"""Plot observed held-out metrics. Grey cells have no eligible runs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from run_sweep import (
    LEVELS, LOSSES, PROTOCOL, RADII, ROOT, RUNS, SEEDS, add_design_arguments,
    base_config, calibration_table, digest, read_results, select_rates, stage_configs,
)


def aggregate(data, kind, placement, metric, loss=None):
    """One mean, sample SD, and n per cell. loss=None is the paired L1−L2 difference."""
    levels = LEVELS[kind]
    shape = len(RADII), len(levels)
    mean, sd, n = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, dtype=int)
    for i, radius in enumerate(RADII):
        for j, level in enumerate(levels):
            cell = (radius, kind if level else "gaussian", level, placement if level else "target_only")
            samples = []
            for seed in SEEDS:
                if loss is None:
                    left, right = (data.get((*cell, name, seed)) for name in ("l1", "l2"))
                    if left is not None and right is not None:
                        samples.append(left[metric] - right[metric])
                else:
                    scores = data.get((*cell, loss, seed))
                    if scores is not None:
                        samples.append(scores[metric])
            if samples:
                mean[i, j], n[i, j] = np.mean(samples), len(samples)
            if len(samples) >= 2:
                sd[i, j] = np.std(samples, ddof=1)
    return mean, sd, n


def color_scale(panels, difference):
    valid = [a[np.isfinite(a)] for a, _, _ in panels if np.isfinite(a).any()]
    data = np.concatenate(valid) if valid else np.array([0., 1.])
    if difference:
        limit = max(abs(data.min()), abs(data.max()), .01)
        return Normalize(-limit, limit)
    lo, hi = data.min(), data.max()
    if hi == lo:
        margin = max(.01, .01 * abs(hi))
        lo, hi = lo - margin, hi + margin
    return Normalize(lo, hi)


def draw_panel(ax, values, cmap_name, norm, kind):
    mean, sd, ns = values
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#dedede")
    view = ax.imshow(mean, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(LEVELS[kind])), [f"{v:g}" for v in LEVELS[kind]])
    ax.set_yticks(range(len(RADII)), RADII)
    for (i, j), count in np.ndenumerate(ns):
        if count == 0:
            continue
        val = mean[i, j]
        label = f"{val:.2f}"
        if count > 1:
            label += f"\n±{sd[i, j]:.2f}"
        if count < len(SEEDS):
            label += f"\nn={count}"
        r, g, b, _ = cmap(norm(val))
        color = "black" if .299*r + .587*g + .114*b > .55 else "white"
        ax.text(j, i, label, color=color, ha="center", va="center", fontsize=8)
    return view


def plot_metrics(data, kind, placement, probe, losses, out):
    psnr = probe == "psnr"
    tasks = ["PSNR (dB)"] if psnr else ["position", "velocity", "future"]
    columns = list(losses)
    if not psnr and {"l1", "l2"} <= set(losses):
        columns.append(None)
    fig, axes = plt.subplots(len(tasks), len(columns), figsize=(4.4 * len(columns), 3.6 * len(tasks) + .6),
                             layout="constrained", squeeze=False)
    heading = "Clean reconstruction PSNR" if psnr else f"{probe.upper()} probe R²"
    fig.suptitle(f"{heading}    |    {kind}, {placement}\n"
                 "Cell: seed mean ± sample SD; n = 3 unless shown. Grey: no observations.",
                 fontsize=13)
    for i, task in enumerate(tasks):
        metric = "psnr" if psnr else f"{task}_r2_{probe}"
        panels = [aggregate(data, kind, placement, metric, loss) for loss in columns]
        shared = color_scale(panels[:len(losses)], difference=False)
        for j, (loss, panel) in enumerate(zip(columns, panels)):
            ax = axes[i, j]
            diff = loss is None
            norm = color_scale([panel], difference=True) if diff else shared
            view = draw_panel(ax, panel, "RdBu_r" if diff else "viridis", norm, kind)
            ax.set_title("L1 − L2 (paired)" if diff else loss.upper())
            ax.set_ylabel(f"{task}\nradius (px)" if j == 0 else "")
            ax.set_xlabel("noise σ" if kind == "gaussian" else "replacement probability")
            if diff or j == len(losses) - 1:
                fig.colorbar(view, ax=ax if diff else list(axes[i, :len(losses)]), shrink=.75)
    file = out / f"{placement}_{kind}_{probe}.png"
    fig.savefig(file, dpi=150)
    plt.close(fig)
    print(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_design_arguments(parser)
    parser.add_argument("--runtime", help="runtime ID if results contain multiple software/device environments")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    base = base_config(args)
    rows = read_results(RUNS)
    matching = {k: v for k, v in rows.items()
                if v["runtime"]["protocol"] == PROTOCOL
                and all(v["config"].get(f) == val for f, val in vars(base).items()
                        if f not in ("radius", "noise_type", "noise_level", "placement", "loss", "seed", "lr"))}
    runtimes = {digest(r["runtime"])[:12]: r["runtime"] for r in matching.values()
                if r["purpose"] == "experiment"}
    if not runtimes:
        print("No completed experimental runs for this design.")
        return
    if args.runtime:
        if args.runtime not in runtimes:
            parser.error(f"Unknown runtime. Available: {runtimes}")
        runtime = runtimes[args.runtime]
    elif len(runtimes) == 1:
        runtime = next(iter(runtimes.values()))
    else:
        parser.error(f"Choose --runtime from {runtimes}")
    table = calibration_table(base, runtime, matching)
    rates = select_rates(table) if table else dict.fromkeys(LOSSES, 1e-3)
    # Exclude all pilots and any timing pilot using a different LR from the selected one.
    from run_sweep import identity
    cases = stage_configs(3, base, rates)
    data = {}
    for c in cases:
        row = matching.get(digest(identity(c, "experiment", runtime)))
        if row:
            data[c.radius, c.noise_type, c.noise_level, c.placement, c.loss, c.seed] = row["metrics"]
    if not data:
        print("No eligible experimental runs at the selected learning rates.")
        return
    args.out.mkdir(exist_ok=True, parents=True)
    placements = {key[3] for key in data}
    for placement in sorted(placements):
        for kind in LEVELS:
            losses_seen = {loss for _, noise, level, place, loss, _ in data if place == placement
                           and (level == 0. or kind == noise)}
            if not losses_seen:
                continue
            # The two primary columns retain their places while a sweep is incomplete.
            columns = [loss for loss in LOSSES if loss in losses_seen or loss in ("l1", "l2")]
            for probe in ("ridge", "mlp", "psnr"):
                plot_metrics(data, kind, placement, probe, columns, args.out)


if __name__ == "__main__":
    main()
