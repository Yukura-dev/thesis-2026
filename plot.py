"""Plot observed held-out metrics. Grey cells have no eligible runs."""
import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from run_sweep import (
    FILTERED_PROTOCOL, LEVELS, LOSSES, MAIN_PROTOCOL, PROTOCOL, RADII, ROOT, RUNS, SEEDS, STAGE0_STEPS, add_design_arguments,
    base_config, calibration_table, chosen_gap, digest, identity, main_rates, read_results, selection_runtime, select_rates, stage_configs,
)


def aggregate(data, kind, placement, metric, loss=None, statistic="mean"):
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
                reducer = {"mean": np.mean, "median": np.median, "success": lambda x: np.mean(np.array(x) > .8)}
                mean[i, j], n[i, j] = reducer[statistic](samples), len(samples)
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


def draw_panel(ax, values, cmap_name, norm, kind, statistic="mean"):
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
        if count > 1 and statistic == "mean":
            label += f"\n±{sd[i, j]:.2f}"
        label += f"\nn={count}"
        r, g, b, _ = cmap(norm(val))
        color = "black" if .299*r + .587*g + .114*b > .55 else "white"
        ax.text(j, i, label, color=color, ha="center", va="center", fontsize=8)
    return view


def plot_metrics(data, kind, placement, probe, losses, out, statistic="mean"):
    psnr = probe == "psnr"
    tasks = ["PSNR (dB)"] if psnr else ["position", "velocity", "future"]
    if statistic != "mean":
        tasks = ["position", "velocity"]
    columns = list(losses)
    if statistic == "mean" and not psnr and {"l1", "l2"} <= set(losses):
        columns.append(None)
    fig, axes = plt.subplots(len(tasks), len(columns), figsize=(4.4 * len(columns), 3.6 * len(tasks) + .6),
                             layout="constrained", squeeze=False)
    heading = "Clean reconstruction PSNR" if psnr else f"{probe.upper()} probe R²"
    description = {"mean": "seed mean ± sample SD", "median": "seed median",
                   "success": "fraction of evaluated seeds with R² > 0.8"}[statistic]
    fig.suptitle(f"{heading}    |    {kind}, {placement}\n"
                 f"Cell: {description}. n = evaluated seeds. Grey: no observations.",
                 fontsize=13)
    for i, task in enumerate(tasks):
        metric = "psnr" if psnr else f"{task}_r2_{probe}"
        panels = [aggregate(data, kind, placement, metric, loss, statistic) for loss in columns]
        shared = Normalize(0, 1) if statistic == "success" else color_scale(panels[:len(losses)], difference=False)
        for j, (loss, panel) in enumerate(zip(columns, panels)):
            ax = axes[i, j]
            diff = loss is None
            norm = color_scale([panel], difference=True) if diff else shared
            view = draw_panel(ax, panel, "RdBu_r" if diff else "viridis", norm, kind, statistic)
            ax.set_title("L1 − L2 (paired)" if diff else loss.upper())
            ax.set_ylabel(f"{task}\nradius (px)" if j == 0 else "")
            ax.set_xlabel("noise σ" if kind == "gaussian" else "replacement probability")
            if diff or j == len(losses) - 1:
                fig.colorbar(view, ax=ax if diff else list(axes[i, :len(losses)]), shrink=.75)
    suffix = "" if statistic == "mean" else "_" + statistic
    file = out / f"{placement}_{kind}_{probe}{suffix}.png"
    fig.savefig(file, dpi=150)
    plt.close(fig)
    print(file)


def main_cases(base, rates, *, shared_lr=False):
    if shared_lr:
        return stage_configs(1, base, rates, shared_lr=True)
    stages = (1, 2) if base.reject_reflections else (1, 2, 3)
    return list(dict.fromkeys(c for stage in stages for c in stage_configs(stage, base, rates)))


def with_reconstruction_metrics(source, observations):
    """Join only the sidecar record minted for this experiment and runtime."""
    from collapse_replay import spec_for
    observed = observations.get(digest(spec_for(source)))
    if observed is None:
        return source["metrics"]
    if (observed["source_config_hash"] != source["config_hash"] or
            observed["model_sha256"] != source["model_sha256"] or
            observed["probe_score_stream"] != [7, 0] or source["probe_splits"] != [6, 7] or
            not observed["original_validation_exact"] or not observed["original_reconstruction_metrics_exact"] or
            observed["metrics"]["psnr"] != source["metrics"]["psnr"]):
        raise ValueError(f"Reconstruction diagnostic does not match its archived fit: {source['config_hash']}")
    fields = ("corrected_psnr", "affine_a", "affine_b", "background_psnr", "output_pixel_sd", "mean_frame_pixel_sd")
    return {**source["metrics"], **{name: observed["metrics"][name] for name in fields}}


def summary_cells(cases, data):
    """Seed-wise values and intended/observed denominators for every unique condition."""
    groups = defaultdict(list)
    for c in cases:
        groups[(c.radius, c.noise_type, c.noise_level, c.placement, c.loss, c.lr)].append(c.seed)
    output = []
    for (r, kind, level, place, loss, lr), seeds in groups.items():
        intended = sorted(set(seeds))
        seeds_done = {seed: data[(r, kind, level, place, loss, seed)] for seed in intended
                      if (r, kind, level, place, loss, seed) in data}
        n = len(seeds_done)
        stats = {}
        for probe in ("ridge", "mlp"):
            for target in ("position", "velocity"):
                metric = f"{target}_r2_{probe}"
                scores = [m[metric] for m in seeds_done.values()]
                passed = sum(value > .8 for value in scores)
                stats[metric] = {"median": float(np.median(scores)) if scores else None,
                                 "successes": passed, "success_rate": passed / n if n else None}
        if any("corrected_psnr" in m for m in seeds_done.values()):
            for metric in ("psnr", "corrected_psnr"):
                scores = [m[metric] for m in seeds_done.values() if metric in m]
                stats[metric] = {"mean": float(np.mean(scores)),
                                 "sd": float(np.std(scores, ddof=1)) if len(scores) >= 2 else None,
                                 "n": len(scores)}
        output.append({"radius": r, "noise_type": kind, "noise_level": level, "placement": place,
                       "loss": loss, "lr": lr, "intended_seeds": intended, "n_evaluated": n,
                       "complete": n == len(intended), "per_seed": seeds_done, "summary": stats})
    return output


def draw_selection_if_complete(rows, runtime, out=None):
    """The interval study has its own real success map even before any main stage is run."""
    directory = out if out is not None else ROOT / "figures"
    gap_runtime = selection_runtime(runtime)
    if runtime["protocol"] == FILTERED_PROTOCOL:
        from filtered_study import selected_gap
        from plot_filtered_gaps import CSV_DESTINATION, DESTINATION, write_outputs
        try:
            selected_gap(rows, gap_runtime)
        except ValueError:
            return False
        path = directory / DESTINATION.name
        csv_path = CSV_DESTINATION if out is None else directory / CSV_DESTINATION.name
        write_outputs(rows, gap_runtime["device"], path, csv_path, runtime=gap_runtime)
        print(f"Filtered gap screen (30 measured fits): {path}")
        return True
    from gap_study import SELECTION, selected_gap
    from summarize_gaps import selection_figure, summaries
    try:
        selected_gap(rows, gap_runtime)
    except ValueError:
        return False
    trials = [r for r in rows.values() if r["purpose"] == SELECTION and r["runtime"] == gap_runtime]
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / "gap_selection_success.png"
    selection_figure(summaries(trials), path)
    print(f"Gap selection only: {path}")
    return True


def main():
    from training import resolve_device
    from run_sweep import runtime_info
    parser = argparse.ArgumentParser(description=__doc__)
    add_design_arguments(parser)
    parser.add_argument("--runtime", help="runtime ID if result rows contain multiple environments")
    parser.add_argument("--historical", action="store_true", help="plot only the original 300-step stage 0")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3),
                        help="limit the plotted design to this stage (filtered stage 3: arm-specific seed sets)")
    parser.add_argument("--shared-lr", action="store_true", help="plot stage 1 shared-LR sensitivity")
    parser.add_argument("--out", type=Path, help="output directory")
    args = parser.parse_args()
    archive = args.historical or (not args.reject_reflections and args.steps == 300 and args.frame_gap is None and not args.shared_lr and args.stage is None)
    if args.reject_reflections is None:
        args.reject_reflections = not archive
    if args.reject_reflections and (archive or args.frame_gap is not None):
        parser.error("filtered plots require the selected frame gap")
    if args.shared_lr and args.stage not in (None, 1):
        parser.error("shared-LR sensitivity is defined only for stage 1")
    if archive and args.stage is not None:
        parser.error("historical stage 0 and stages 1–3 use different probe splits")
    if archive and (args.steps not in (None, 300) or args.frame_gap not in (None, 1) or args.shared_lr):
        parser.error("historical stage 0 uses 300 steps, frame gap 1 and the selected L2 rate")
    if archive:
        args.steps = 300
    base = base_config(args)
    rows = read_results(RUNS)
    protocol = FILTERED_PROTOCOL if args.reject_reflections else PROTOCOL if archive else MAIN_PROTOCOL
    runtimes = {digest(r["runtime"])[:12]: r["runtime"] for r in rows.values()
                if r["runtime"]["protocol"] == protocol and r["purpose"] == "experiment"}
    if not archive:
        selector = selection_runtime(runtime_info(resolve_device(), filtered=args.reject_reflections))
        purpose = "filtered_gap_selection" if args.reject_reflections else "gap_selection"
        for row in rows.values():
            if row["purpose"] == purpose and row["runtime"]["protocol"] == selector["protocol"]:
                final_runtime = {**row["runtime"], "protocol": protocol}
                runtimes[digest(final_runtime)[:12]] = final_runtime
    if args.runtime is not None:
        if args.runtime not in runtimes:
            parser.error(f"Unknown runtime. Available: {runtimes}")
        runtime = runtimes[args.runtime]
    elif len(runtimes) > 1:
        parser.error(f"Choose --runtime from {runtimes}")
    else:
        runtime = next(iter(runtimes.values()), runtime_info(resolve_device(), historical=archive, filtered=args.reject_reflections))
    drew_selection = False
    if not archive and not args.shared_lr and args.frame_gap is None and args.stage is None:
        drew_selection = draw_selection_if_complete(rows, runtime, args.out)
    if archive:
        table = calibration_table(base, runtime, rows)
        rates = select_rates(table) if table else dict.fromkeys(LOSSES, 1e-3)
        cases = stage_configs(0, base, rates)
    else:
        try:
            base = replace(base, frame_gap=chosen_gap(rows, runtime, args.frame_gap))
            rates = dict.fromkeys(LOSSES, 1e-3) if args.shared_lr else main_rates(base, runtime, rows)
        except (ValueError, FileNotFoundError) as exc:
            if drew_selection:
                print(f"No main-stage heatmaps: {exc}")
                return
            parser.error(str(exc))
        cases = (stage_configs(args.stage, base, rates, shared_lr=args.shared_lr)
                 if args.stage is not None else main_cases(base, rates, shared_lr=args.shared_lr))
    out = args.out
    if out is None:
        out = ROOT / "figures"
        if args.reject_reflections:
            out /= "filtered"
        if not archive:
            out = out / f"steps_{base.steps}" / f"gap_{base.frame_gap}"
            if (base.latent_dim, base.background, base.foreground) != (16, .3, 1.):
                out /= f"z_{base.latent_dim}_bg_{base.background:g}_fg_{base.foreground:g}"
            out /= "shared_lr" if args.shared_lr else "main"
            if args.stage is not None:
                out /= f"stage_{args.stage}"
    if args.stage == 3 and runtime["protocol"] == FILTERED_PROTOCOL:
        from stage3_summary import write_outputs
        write_outputs(rows, runtime, base, rates, out)
        return
    observations = {}
    if runtime["protocol"] == FILTERED_PROTOCOL:
        from collapse_replay import LEDGER
        observations = read_results(LEDGER)
    data = {}
    for c in cases:
        row = rows.get(digest(identity(c, "experiment", runtime)))
        if row:
            metrics = with_reconstruction_metrics(row, observations) if observations else row["metrics"]
            data[c.radius, c.noise_type, c.noise_level, c.placement, c.loss, c.seed] = metrics
    if not data:
        print("No eligible main-stage runs; no main-stage heatmaps generated.")
        return
    out.mkdir(exist_ok=True, parents=True)
    cells = summary_cells(cases, data)
    report = {"latent_dim": base.latent_dim, "background": base.background, "foreground": base.foreground,
              "reject_reflections": base.reject_reflections, "frame_gap": base.frame_gap, "steps": base.steps, "runtime": runtime,
              "stage": args.stage, "shared_lr": args.shared_lr, "success_rule": "R² > 0.8", "conditions": cells}
    n_corrected = sum("corrected_psnr" in m for m in data.values())
    if n_corrected:
        report["reconstruction_diagnostics"] = {
            "source": str(LEDGER.relative_to(ROOT)), "n_evaluated": n_corrected,
            "corrected_psnr": "One scalar affine fit per run across all clean test pixels in both frames.",
        }
    (out / "seed_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + "\n")
    for placement in sorted({key[3] for key in data}):
        for kind in LEVELS:
            losses_seen = {loss for _, noise, level, place, loss, _ in data if place == placement
                           and (level == 0. or kind == noise)}
            if not losses_seen:
                continue
            columns = [loss for loss in LOSSES if loss in losses_seen or loss in ("l1", "l2")]
            for probe in ("ridge", "mlp", "psnr"):
                plot_metrics(data, kind, placement, probe, columns, out)
                if probe != "psnr" and not archive:
                    for stat in ("median", "success"):
                        plot_metrics(data, kind, placement, probe, columns, out, statistic=stat)


if __name__ == "__main__":
    main()
