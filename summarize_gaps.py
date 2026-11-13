"""Write seed-level gap selection and pilot summaries from the measured journal."""
import csv
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gap_study import (CALIBRATION, GAP_RADII, GAPS, SELECTION, runtime_info,
                       selected_gap, selected_rates)
from run_sweep import ROOT, read_results


def trial_key(row):
    c = row["config"]
    return c["frame_gap"], c["radius"], c["loss"], c["lr"]


def summaries(rows):
    groups = {}
    for row in rows:
        groups.setdefault(trial_key(row), []).append(row)
    result = []
    for (gap, radius, loss, lr), trials in sorted(groups.items()):
        trials.sort(key=lambda row: row["config"]["seed"])
        for probe in ("mlp", "ridge"):
            for task in ("velocity", "position"):
                scores = [r["metrics"][f"{task}_r2_{probe}"] for r in trials]
                n = len(scores)
                count = sum(value > 0.8 for value in scores)
                result.append({"frame_gap": gap, "radius": radius, "loss": loss, "lr": lr,
                               "probe": probe, "target": task, "n": n,
                               "seed_ids": " ".join(str(r["config"]["seed"]) for r in trials),
                               "seed_r2": " ".join(repr(v) for v in scores),
                               "mean_r2": sum(scores) / n, "median_r2": median(scores),
                               "successes": count, "success_rate": count / n})
    return result


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def selection_figure(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, ("position", "velocity")):
        scores = np.full((len(GAP_RADII), len(GAPS)), np.nan)
        for row in summary:
            if row["probe"] == "mlp" and row["target"] == task:
                i, j = GAP_RADII.index(row["radius"]), GAPS.index(row["frame_gap"])
                scores[i, j] = row["success_rate"]
                ax.text(j, i, f"{row['successes']}/{row['n']}", ha="center", va="center",
                        fontsize=13, color="white" if row["success_rate"] < 0.55 else "black")
        im = ax.imshow(scores, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(task.capitalize() + " MLP")
        ax.set_xticks(range(len(GAPS)), GAPS)
        ax.set_yticks(range(len(GAP_RADII)), GAP_RADII)
        ax.set_xlabel("Frame gap (frames)")
    axes[0].set_ylabel("Radius (px)")
    fig.colorbar(im, ax=axes, label="Fraction of seeds", ticks=[0, .2, .4, .6, .8, 1],
                 shrink=.9)
    fig.suptitle("Seeds with test R² > 0.8   |   clean L2, 3,000 steps")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto","mps","cuda","cpu"), default="auto")
    args = parser.parse_args()
    runtime = runtime_info(args.device)
    rows = read_results()
    chosen = selected_gap(rows, runtime)  # A partial grid cannot produce the final summaries.
    selection = [r for r in rows.values() if r["runtime"] == runtime and r["purpose"] == SELECTION]
    results = ROOT / "results"
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    write_csv(results / "gap_selection_seeds.csv", [
        {"frame_gap":r["config"]["frame_gap"], "radius":r["config"]["radius"], "seed":r["config"]["seed"],
         **r["metrics"], "temporal_rms_ratio":r["temporal_reconstruction"]["rms_ratio"],
         "source_config_hash":r.get("reused_from",r["config_hash"])}
        for r in sorted(selection, key=lambda r: (trial_key(r),r["config"]["seed"]))])
    summary = summaries(selection)
    write_csv(results / "gap_selection_summary.csv", summary)
    selection_figure(summary, figures / "gap_selection_success.png")
    pilots = [r for r in rows.values() if r["runtime"] == runtime and r["purpose"] == CALIBRATION
              and r["config"]["frame_gap"] == chosen]
    if pilots:
        selected_rates(rows, runtime, chosen)
        write_csv(results / "gap_lr_summary.csv", summaries(pilots))
    print(f"Selected gap: {chosen}; wrote {len(selection)} seed rows and {len(summary)} summaries.")


if __name__ == "__main__":
    main()
