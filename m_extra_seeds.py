"""M extra seeds: more seeds at the conditions where noisy/gauss/smooth L1 differ.

Conditions (r, sigma): (5, 0.2), (8, 0.02), (8, 0.05); arms noisy_l1, gauss_l1, smooth_l1;
new seeds 5-19 (seeds 0-4 are the main M runs). Same protocol, ledger and evaluation as M,
so rows are directly comparable; the main M export still uses seeds 0-4 only.
"""
import argparse
import csv
import math
import statistics
import time
from collections import defaultdict

from run_sweep import digest, read_results
from smoothing_study import (OUT, arm_config, classify, experiment_spec, resolve_device, run_one,
                             runtime_info)

CONDITIONS = ((5, .2), (8, .02), (8, .05))
ARMS = ("noisy_l1", "gauss_l1", "smooth_l1")
NEW_SEEDS = range(5, 20)
ALL_SEEDS = range(20)
EXTRA_OUT = OUT.parent / "final_m_extra"


def plan(seeds=NEW_SEEDS):
    return [(arm, r, s, seed, arm_config(arm, r, s, seed))
            for (r, s) in CONDITIONS for seed in seeds for arm in ARMS]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def export(rows, runtime):
    records = []
    for arm, r, s, seed, cfg in plan(ALL_SEEDS):
        key = digest(experiment_spec(cfg, runtime))
        if key in rows:
            m = rows[key]["metrics"]
            records.append({"radius": r, "sigma": s, "arm": arm, "seed": seed, "class": classify(m),
                            "velocity_r2_mlp": m["velocity_r2_mlp"], "psnr": m["psnr"],
                            "success": m["velocity_r2_mlp"] > .8, "model_sha256": rows[key]["model_sha256"]})
    EXTRA_OUT.mkdir(parents=True, exist_ok=True)
    with (EXTRA_OUT / "seeds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    by = defaultdict(dict)
    for x in records:
        by[(x["radius"], x["sigma"], x["arm"])][x["seed"]] = x
    lines = ["| r | σ | arm | n | 成功 (95% Wilson) | 両フレーム | 背景/過去/現在/両方 | 速度R² 平均 | PSNR 平均 |",
             "|---:|---:|---|---:|---|---:|---|---:|---:|"]
    for (r, s) in CONDITIONS:
        for arm in ARMS:
            g = list(by[(r, s, arm)].values())
            if not g:
                continue
            k = sum(x["success"] for x in g)
            lo, hi = wilson(k, len(g))
            counts = "/".join(str(sum(x["class"] == c for x in g)) for c in ("background", "past", "current", "both"))
            lines.append(f"| {r} | {s:g} | {arm} | {len(g)} | {k}/{len(g)} ({lo:.2f}–{hi:.2f}) | "
                         f"{sum(x['class'] == 'both' for x in g)} | {counts} | "
                         f"{statistics.fmean(x['velocity_r2_mlp'] for x in g):.3f} | "
                         f"{statistics.fmean(x['psnr'] for x in g):.2f} |")
    lines += ["", "| r | σ | 比較 | 共通シード | 成否一致 | 両群で混在? | 片方だけ成功 (noisy, other) |", "|---:|---:|---|---:|---:|---|---|"]
    for (r, s) in CONDITIONS:
        for other in ("gauss_l1", "smooth_l1"):
            a, b = by[(r, s, "noisy_l1")], by[(r, s, other)]
            common = sorted(set(a) & set(b))
            if not common:
                continue
            agree = sum(a[i]["success"] == b[i]["success"] for i in common)
            only_a = sum(a[i]["success"] and not b[i]["success"] for i in common)
            only_b = sum(b[i]["success"] and not a[i]["success"] for i in common)
            mixed = all(0 < sum(d[i]["success"] for i in common) < len(common) for d in (a, b))
            lines.append(f"| {r} | {s:g} | noisy vs {other} | {len(common)} | {agree}/{len(common)} | "
                         f"{'yes' if mixed else 'no'} | {only_a}, {only_b} |")
    (EXTRA_OUT / "tables.md").write_text(
        "# M 追加シード（しきい値付近）\n\nシード0〜19。シード0〜4は実験M本体と同じ学習。成功は速度MLP R² > 0.8。\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    runtime = runtime_info(resolve_device(args.device))
    cases = plan()
    rows = read_results()
    done = sum(digest(experiment_spec(c[-1], runtime)) in rows for c in cases)
    print(f"M extra: {len(cases)} runs, {done} complete on {runtime['device']}", flush=True)
    if not args.export_only:
        start = time.perf_counter()
        for i, (*_, cfg) in enumerate(cases, 1):
            run_one(cfg, runtime)
            if i % 9 == 0:
                print(f"M extra progress {i}/{len(cases)} after {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    print(f"Exported {export(read_results(), runtime)} rows to {EXTRA_OUT}", flush=True)


if __name__ == "__main__":
    main()
