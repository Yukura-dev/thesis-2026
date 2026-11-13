"""Plot paired seeds only when all 30 clean, filtered screen fits are present."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import filtered_study as study
from gap_study import VELOCITY_THRESHOLD, _complete_rows
from run_sweep import ROOT, read_results

DESTINATION = ROOT / "figures" / "filtered_gap_screen.png"
CSV_DESTINATION = ROOT / "results" / "filtered_gap_seeds.csv"
FIELDS = ("frame_gap", "radius", "seed", "position_r2_mlp", "velocity_r2_mlp", "position_r2_ridge",
          "velocity_r2_ridge", "oracle_velocity_r2_mlp", "temporal_rms_ratio", "center_intensity",
          "center_contrast", "psnr", "config_hash")
COLORS = ("#006FA1", "#D05A24", "#18816D", "#8C63A6", "#735736")
MARKERS = ("o", "s", "^", "D", "v")


def collect(rows, device, *, runtime=None):
    runtime = study.runtime_info(device) if runtime is None else runtime
    centers = study._centers()
    measured = _complete_rows(study.selection_configs(), study.SELECTION, rows, runtime)
    samples = []
    for row in measured:
        cfg = row["config"]
        gap, radius, seed = cfg["frame_gap"], cfg["radius"], cfg["seed"]
        if row["probe_fit_stream"] != [2, 0] or row["probe_score_stream"] != [3, 0]:
            raise ValueError("Filtered AE probe sample IDs differ from the oracle")
        if row["true_center_velocity_r2_mlp"] != centers[gap, radius]:
            raise ValueError(f"Oracle mismatch: gap {gap}, radius {radius}, seed {seed}")
        sample = {k: cfg[k] for k in ("frame_gap", "radius", "seed")}
        sample.update({k: row["metrics"][k] for k in ("position_r2_mlp", "velocity_r2_mlp",
                       "position_r2_ridge", "velocity_r2_ridge", "center_intensity", "center_contrast", "psnr")})
        sample["oracle_velocity_r2_mlp"] = centers[gap, radius]
        sample["temporal_rms_ratio"] = row["temporal_reconstruction"]["rms_ratio"]
        sample["config_hash"] = row["config_hash"]
        if not all(np.isfinite(sample[k]) for k in FIELDS if k != "config_hash"):
            raise ValueError(f"Nonfinite measurement: gap {gap}, radius {radius}, seed {seed}")
        samples.append(sample)
    samples.sort(key=lambda s: (s["frame_gap"], s["radius"], s["seed"]))
    return samples, centers


def render(values, centers, output):
    """Values are used at their measured heights; offsets only separate seed markers on x."""
    limits = []
    for index in (0, 1):
        vals = [v[index] for v in values.values()]
        bottom, top = min(min(vals), 0.), max(max(vals), 1.)
        padding = .07 * (top - bottom)
        limits.append((bottom-padding, top+padding))
    with plt.rc_context({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.right": False, "axes.spines.top": False}):
        fig, axes = plt.subplots(2, 3, figsize=(10.7, 6.4), sharex=True, sharey="row")
        fig.subplots_adjust(left=.105, right=.98, top=.82, bottom=.22, wspace=.15, hspace=.25)
        fig.text(.105, .955, "Velocity readout and reconstructed motion", size=19,
                 fontweight="bold", ha="left", color="#22313A")
        fig.text(.105, .909, "No observed wall hits   ·   3,000 L2 updates   ·   Five training seeds per radius",
                 size=11, color="#42535B")
        legend_handles = []
        for j, radius in enumerate(study.RADII):
            upper, lower = axes[:, j]
            upper.set_title(f"Radius {radius} px", loc="left", size=12, fontweight="bold", pad=8)
            for i, ax in enumerate((upper, lower)):
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#E3E7E9", linewidth=.75)
                ax.spines[["left", "bottom"]].set_color("#9AA9AE")
                ax.tick_params(length=3, color="#829296")
                ax.set_xlim(1.60, 4.40)
                ax.set_ylim(*limits[i])
                ax.set_xticks((2, 4), ("2 frames", "4 frames"))
            upper.axhline(VELOCITY_THRESHOLD, ls="--", lw=1.25, color="#52606A", zorder=1)
            upper.plot((2, 4), [centers[g, radius] for g in (2, 4)], marker="X", ls=":",
                       lw=1.15, color="#232A2F", markersize=6., zorder=5)
            lower.axhline(1., lw=1.15, ls=":", color="#52606A", zorder=1)
            for seed, (color, marker) in enumerate(zip(COLORS, MARKERS)):
                xs = np.array((2., 4.)) + .044 * (seed-2)
                for i, ax in enumerate((upper, lower)):
                    ys = [values[g, radius, seed][i] for g in (2, 4)]
                    ax.plot(xs, ys, marker=marker, lw=1.2, markersize=5.8, color=color,
                            markeredgecolor="white", markeredgewidth=.55, alpha=.88, zorder=3)
                if j == 0:
                    legend_handles.append(Line2D([], [], color=color, marker=marker, lw=1.3, label=f"Seed {seed}"))
        axes[0, 0].set_ylabel("Velocity MLP  R²", fontsize=11)
        axes[1, 0].set_ylabel("Temporal RMS ratio", fontsize=11)
        fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(.102, .104),
                   ncol=5, frameon=False, borderaxespad=0, handlelength=1.6, columnspacing=1.8)
        fig.text(.105, .08, "Upper: black crosses = true-center MLP; dashed line = 0.80 reliability threshold.",
                 size=9, color="#42535B")
        fig.text(.105, .049, "Lower: reconstructed / clean frame-difference RMS; dotted line = equal amplitude.",
                 size=9, color="#42535B")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as tmp:
                fig.savefig(tmp, format="png", dpi=180, facecolor="white",
                            metadata={"Title": "Velocity readout under reflection-free observation windows"})
            os.replace(tmp.name, output)
        finally:
            plt.close(fig)


def csv_text(samples):
    """Canonical report columns and full-precision Python float representation."""
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(samples)
    return buffer.getvalue()


def write_outputs(rows, device="auto", output=DESTINATION, csv_output=CSV_DESTINATION, *, runtime=None):
    """Rebuild both artifacts from exactly the complete matched screening grid."""
    output, csv_output = Path(output), Path(csv_output)
    if csv_output.resolve() == output.resolve():
        raise ValueError("Figure and CSV must have distinct paths")
    samples, centers = collect(rows, device, runtime=runtime)
    values = {(s["frame_gap"], s["radius"], s["seed"]): (s["velocity_r2_mlp"], s["temporal_rms_ratio"])
              for s in samples}
    render(values, centers, output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=csv_output.parent, delete=False) as tmp:
        tmp.write(csv_text(samples))
    os.replace(tmp.name, csv_output)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--csv", type=Path, default=CSV_DESTINATION)
    args = parser.parse_args()
    try:
        samples = write_outputs(read_results(), args.device, args.output, args.csv)
    except ValueError as exc:
        parser.exit(2, f"No output generated: {exc}\n")
    print(f"Saved {args.output} and {args.csv} from {len(samples)} measured runs.")


if __name__ == "__main__":
    main()
