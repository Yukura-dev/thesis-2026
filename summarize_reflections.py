"""Summarize matched original-model, true-center and Bayes velocity predictions."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gap_study import GAPS, GAP_RADII, SELECTION, selection_configs, study_spec
from reflection_study import PROTOCOL, PURPOSE
from run_sweep import ROOT, RUNS, digest, read_results

GROUPS = ("all", "reflecting", "nonreflecting")


def collect(rows, oracle, device):
    """Verify a complete paired design before returning any data for publication."""
    rts = {json.dumps(r["runtime"], sort_keys=True) for r in rows.values()
           if r["purpose"] == PURPOSE and r["runtime"]["protocol"] == PROTOCOL
           and (device is None or r["device"] == device)
           and all(r["runtime"][lib] == oracle["runtime"][lib] for lib in ("numpy", "torch"))}
    if len(rts) != 1:
        raise ValueError(f"Expected one matching stratification runtime; found {len(rts)}. Check --device and oracle.")
    runtime = json.loads(rts.pop())
    original_runtime = {**runtime, "protocol": "circle-ae-gap-v1"}
    centers = {(c["frame_gap"], c["radius"]): c for c in oracle["conditions"]}
    if (len(oracle["conditions"]) != len(GAPS) * len(GAP_RADII)
            or set(centers) != {(d, r) for d in GAPS for r in GAP_RADII}):
        raise ValueError("Incomplete or duplicate true-center oracle conditions")
    if (oracle["seed_namespaces"] != {"fit": [2, 0], "test": [3, 0]}
            or (oracle["fit_n"], oracle["test_n"]) != (10000, 5000)):
        raise ValueError("Oracle and original latent probes must use the same 10k/5k examples")

    result = []
    configs = selection_configs()
    missing = sum(digest(study_spec(c, PURPOSE, runtime)) not in rows for c in configs)
    if missing:
        raise ValueError(f"Incomplete comparison: {missing} of {len(configs)} model readouts missing")
    for c in configs:
        key = digest(study_spec(c, PURPOSE, runtime))
        measured = rows[key]
        source_key = digest(study_spec(c, SELECTION, original_runtime))
        source = rows.get(source_key)
        if (source is None or measured["source_config_hash"] != source_key
                or measured["model_sha256"] != source["model_sha256"]
                or measured["ten_metrics_exact_match"] is not True or measured["metrics"] != source["metrics"]
                or source["probe_fit_stream"] != [2, 0] or source["probe_score_stream"] != [3, 0]):
            raise ValueError(f"Original source/weights/scores do not match: {key}")
        reference = centers[c.frame_gap, c.radius]
        counts = reference["counts"]
        if (counts["total"] != c.probe_test or measured["n_test"] != c.probe_test
                or counts["reflecting"] != measured["n_reflecting"]
                or counts["nonreflecting"] + counts["reflecting"] != counts["total"]):
            raise ValueError(f"Oracle and latent subsets do not match: {key}")
        for group in GROUPS:
            count = counts["total" if group == "all" else group]
            methods = {"latent_mlp": measured["stratified"]["mlp"][group],
                       "latent_ridge": measured["stratified"]["ridge"][group],
                       "true_center_mlp": reference["oracle_mlp"][group],
                       "bayes_test": reference["bayes_endpoint_mean"][group]}
            if any(stats["n"] != count or not math.isfinite(stats["r2"]) for stats in methods.values()):
                raise ValueError(f"Invalid held-out subset/score: {key}, {group}")
            if group == "all" and any(
                    not math.isclose(methods[f"latent_{p}"]["r2"], source["metrics"][f"velocity_r2_{p}"],
                                     rel_tol=0, abs_tol=1e-12) for p in ("mlp", "ridge")):
                raise ValueError(f"All-pair R² differs from the original probe: {source_key}")
            result.append({"frame_gap": c.frame_gap, "radius": c.radius, "seed": c.seed,
                           "subset": group, "n": count, "fraction_of_test": count / c.probe_test,
                           **{f"{name}_r2": score["r2"] for name, score in methods.items()},
                           "population_bayes_r2": reference["population_bayes_r2"] if group == "all" else None,
                           "source_config_hash": source_key, "stratification_config_hash": key,
                           "source_model_sha256": measured["model_sha256"],
                           "weight_provenance": measured["weight_provenance"]})
    return result


def plot(rows, path):
    names = ("All trajectories", "Reflected", "No reflection")
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True, sharey=True)
    for i, (group, name) in enumerate(zip(GROUPS, names)):
        for j, radius in enumerate(GAP_RADII):
            ax = axes[i, j]
            groups = [[r for r in rows if (r["subset"], r["radius"], r["frame_gap"]) == (group, radius, gap)]
                      for gap in GAPS]
            xs = range(len(GAPS))
            for x, by_seed in zip(xs, groups):
                ax.scatter([x + (r["seed"]-2)*.032 for r in by_seed],
                           [r["latent_mlp_r2"] for r in by_seed], color="#5879ad", s=14, alpha=.55,
                           label="Latent MLP (each seed)" if x == 0 else None, zorder=3)
            ax.plot(xs, [median(r["latent_mlp_r2"] for r in g) for g in groups],
                    "o-", color="#1c4b88", ms=5, label="Latent MLP (median of 5)", zorder=4)
            ax.plot(xs, [g[0]["true_center_mlp_r2"] for g in groups],
                    "s-", color="#b34e30", ms=4, label="True-center MLP")
            ax.plot(xs, [g[0]["bayes_test_r2"] for g in groups],
                    "^--", color="#37856f", ms=4, label="Bayes (same test pairs)")
            if group == "all":
                ax.plot(xs, [g[0]["population_bayes_r2"] for g in groups],
                        ":", color="#26262f", lw=2, label="Bayes (population, all pairs)")
            ax.axhline(.8, color="#a4a4a4", lw=.8, ls="--", zorder=0)
            ax.set_xticks(xs, [str(g) for g in GAPS])
            ax.grid(axis="y", color="#e3e5e8", lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"Radius {radius} px")
            if j == 0:
                ax.set_ylabel(f"{name}\nVelocity R²")
            if i == 2:
                ax.set_xlabel("Frame gap")
    fig.suptitle("Velocity from two observed frames: original reflecting trajectories", y=.98, fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=3,
               fontsize=9, frameon=False)
    fig.subplots_adjust(top=.80, left=.09, right=.985, bottom=.08, hspace=.12, wspace=.12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), help="select recorded latent run device")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--oracle", type=Path, default=ROOT / "results" / "reflection_oracle.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "reflection_stratification_seeds.csv")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures" / "reflection_velocity.png")
    args = parser.parse_args()
    try:
        records = collect(read_results(args.runs), json.loads(args.oracle.read_text()), args.device)
    except ValueError as exc:
        parser.error(str(exc))
    args.csv.parent.mkdir(exist_ok=True, parents=True)
    args.figure.parent.mkdir(exist_ok=True, parents=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plot(records, args.figure)
    print(f"{len(records)} seed/subset rows: {args.csv}; figure: {args.figure}")


if __name__ == "__main__":
    main()
