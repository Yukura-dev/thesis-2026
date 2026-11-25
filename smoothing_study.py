"""M: is L1 background collapse caused by the kink of |e| at zero error?

Hypothesis. A decoder that outputs the exact background makes almost every pixel error
exactly zero. Under L1 any z-dependent change of the output then pays a first-order
cost |delta| on many background pixels, so the constant output is a sharp (nonsmooth)
trap. Gaussian target noise of SD sigma replaces |e| in expectation by the smooth
E|e+n| (curvature sqrt(2/pi)/sigma at e=0), which removes the trap.

Arms (all r=5,8, seeds 0-4, Delta=4, lr 0.003, 3,000 steps, clean input):
  noisy_l1   L1 with Gaussian target noise sigma (sigma=0 is plain L1)      - re-measured control
  gauss_l1   clean target, loss = analytic E|e+n| at the same sigma         - same expected objective,
                                                                               no gradient stochasticity
  smooth_l1  clean target, unit-slope Huber with beta = sigma*sqrt(pi/2)    - same curvature at 0,
                                                                               different tail shape
  l2         clean reference
Predictions: if smoothing is the mechanism, gauss_l1 ~ noisy_l1 at each sigma (and smooth_l1
also, if only local curvature matters). If stochastic gradients are the mechanism, gauss_l1
and smooth_l1 collapse like plain L1.
"""
import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import replace

from filelock import FileLock

from run_sweep import ROOT, RUNS, config_record, digest, read_results, runtime_info as main_runtime
from training import Config, resolve_device, run_experiment

PURPOSE = "final_m_loss_smoothing"
PROTOCOL = "circle-ae-loss-smoothing-v1"
SPLITS = (6, 7)
SIGMAS = (.02, .05, .1, .2, .4)
RADII = (5, 8)
SEEDS = range(5)
THRESHOLD = .65
GPU_LOCK = ROOT / "checkpoints" / "final_m_gpu.lock"
OUT = RUNS.parent / "final_m"
BETA_PER_SIGMA = math.sqrt(math.pi / 2)  # equal curvature at zero: 1/beta = sqrt(2/pi)/sigma


def base():
    return Config(frame_gap=4, steps=3000, lr=.003, reject_reflections=True)


def arm_config(arm, radius, sigma, seed):
    cfg = replace(base(), radius=radius, seed=seed)
    if arm == "noisy_l1":
        return replace(cfg, loss="l1", noise_type="gaussian", noise_level=sigma)
    if arm == "gauss_l1":
        return replace(cfg, loss="gauss_l1", loss_scale=sigma)
    if arm == "smooth_l1":
        return replace(cfg, loss="smooth_l1", loss_scale=sigma * BETA_PER_SIGMA)
    if arm == "l2":
        return replace(cfg, loss="l2")
    raise ValueError(arm)


def plan():
    """(arm, radius, sigma, seed, config), ordered so each sigma's three arms finish together."""
    cases = [("l2", r, 0., s) for r in RADII for s in SEEDS]
    cases += [("noisy_l1", r, 0., s) for r in RADII for s in SEEDS]
    for sigma in SIGMAS:
        for r in RADII:
            for arm in ("noisy_l1", "gauss_l1", "smooth_l1"):
                cases += [(arm, r, sigma, s) for s in SEEDS]
    return [(arm, r, sigma, s, arm_config(arm, r, sigma, s)) for arm, r, sigma, s in cases]


def runtime_info(device):
    return {**main_runtime(device, filtered=True), "protocol": PROTOCOL}


def experiment_spec(config, runtime):
    return {"config": config_record(config), "purpose": PURPOSE, "runtime": runtime}


def classify(metrics):
    past = metrics["center_intensity_previous"] >= THRESHOLD
    current = metrics["center_intensity_current"] >= THRESHOLD
    return {(False, False): "background", (True, False): "past", (False, True): "current",
            (True, True): "both"}[(past, current)]


def _append(path, row):
    with path.open("a") as out:
        out.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())


def run_one(config, runtime, path=RUNS, gpu_lock=GPU_LOCK, store=True):
    spec = experiment_spec(config, runtime)
    key = digest(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        saved = read_results(path, repair=True).get(key)
    if saved is not None:
        return saved
    with FileLock(str(gpu_lock)):
        with FileLock(str(path) + ".lock"):
            saved = read_results(path, repair=True).get(key)
        if saved is not None:
            return saved
        print(f"run M {key[:10]} r={config.radius} {config.loss} scale={config.loss_scale:g} "
              f"noise={config.noise_level:g} seed={config.seed}", flush=True)
        persist = None
        if store:
            import torch
            from diagnose_steps import save_checkpoint
            cache = ROOT / "checkpoints" / "final_m" / key / "weights.pt"

            def persist(model):
                save_checkpoint(cache, {"source_config_hash": key,
                                        "model": {n: t.detach().cpu() for n, t in model.state_dict().items()}})
        result = run_experiment(config, runtime["device"], probe_splits=SPLITS, affine=True,
                                store_weights=persist)
        row = {**spec, "config_hash": key, "first_stage": None,
               "training_seed_sequence": [0, config.seed], **result}
        with FileLock(str(path) + ".lock"):
            rows = read_results(path, repair=True)
            if key not in rows:
                _append(path, row)
            else:
                row = rows[key]
        m = row["metrics"]
        print(f"saved M {key[:10]} class={classify(m)} PSNR={m['psnr']:.3f} "
              f"velocity={m['velocity_r2_mlp']:.4f} in {row['timing']['total_seconds']:.1f}s", flush=True)
    return row


def historical_reference(rows):
    """Earlier MPS measurements of the same noisy-L1 / L2 configurations (stage 1-3 'experiment')."""
    found = {}
    for row in rows.values():
        if row.get("purpose") != "experiment" or row["runtime"].get("protocol") != "circle-ae-filtered-final-v1":
            continue
        c = row["config"]
        if c.get("placement") != "target_only" or c.get("foreground") != 1.0 or c.get("steps") != 3000:
            continue
        level = c["noise_level"]
        if level and c["noise_type"] != "gaussian":
            continue
        arm = {"l1": "noisy_l1", "l2": "l2"}.get(c["loss"])
        if not arm or (arm == "l2" and level):
            continue
        # Same configuration as the M arm in every recorded field (lr, Delta, batch, probes, ...).
        expected = arm_config(arm, c["radius"], float(level), c["seed"])
        if all(getattr(expected, k) == v for k, v in c.items()):
            found[(arm, c["radius"], float(level), c["seed"])] = row
    return found


def seed_records(rows, runtime):
    records = []
    history = historical_reference(rows)
    for arm, r, sigma, seed, cfg in plan():
        key = digest(experiment_spec(cfg, runtime))
        if key not in rows:
            continue
        row, m = rows[key], rows[key]["metrics"]
        old = history.get((arm, r, sigma, seed))
        records.append({
            "arm": arm, "radius": r, "sigma": sigma, "loss": cfg.loss, "loss_scale": cfg.loss_scale,
            "seed": seed, "class": classify(m), "psnr": m["psnr"], "corrected_psnr": m["corrected_psnr"],
            "center_previous": m["center_intensity_previous"], "center_current": m["center_intensity_current"],
            "output_pixel_sd": m["output_pixel_sd"],
            **{f"{t}_r2_mlp": m[f"{t}_r2_mlp"] for t in ("position", "velocity", "future")},
            "mps_velocity_r2_mlp": old["metrics"]["velocity_r2_mlp"] if old else None,
            "mps_center_intensity": old["metrics"].get("center_intensity") if old else None,
            "validation_loss": row["validation_loss"], "train_seconds": row["timing"]["train_seconds"],
            "model_sha256": row["model_sha256"], "config_hash": key})
    return records


def _stat(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return f"{mean:.3f} ± {sd:.3f} / {statistics.median(values):.3f}"


def summarize(records):
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["radius"], rec["sigma"], rec["arm"])].append(rec)
    order = {"l2": 0, "noisy_l1": 1, "gauss_l1": 2, "smooth_l1": 3}
    lines = ["| r | σ | arm | n | 背景/過去/現在/両方 | PSNR | 補正PSNR | 過去中心 | 現在中心 | 位置R² | 速度R² (成功) | 既存同一設定の速度成功 |",
             "|---:|---:|---|---:|---|---|---|---|---|---|---|---|"]
    table = []
    for (r, sigma, arm) in sorted(groups, key=lambda k: (k[0], k[1], order[k[2]])):
        g = groups[(r, sigma, arm)]
        counts = "/".join(str(sum(x["class"] == c for x in g)) for c in ("background", "past", "current", "both"))
        vel = [x["velocity_r2_mlp"] for x in g]
        ok = sum(v > .8 for v in vel)
        old = [x["mps_velocity_r2_mlp"] for x in g if x["mps_velocity_r2_mlp"] is not None]
        old_ok = f"{sum(v > .8 for v in old)}/{len(old)}" if old else "—"
        lines.append(f"| {r} | {sigma:g} | {arm} | {len(g)} | {counts} | {_stat([x['psnr'] for x in g])} | "
                     f"{_stat([x['corrected_psnr'] for x in g])} | {_stat([x['center_previous'] for x in g])} | "
                     f"{_stat([x['center_current'] for x in g])} | {_stat([x['position_r2_mlp'] for x in g])} | "
                     f"{_stat(vel)} ({ok}/{len(g)}) | {old_ok} |")
        table.append({"radius": r, "sigma": sigma, "arm": arm, "n": len(g), "classes": counts,
                      "velocity_success": ok, "mps_velocity_success": old_ok})
    return "\n".join(lines) + "\n", table


def _write_csv(path, records):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def export(rows, runtime, out=OUT):
    records = seed_records(rows, runtime)
    if not records:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "seeds.csv", records)
    markdown, table = summarize(records)
    header = (f"# M 損失の平滑化\n\n{len(records)}/{len(plan())} 設定。各セルは平均 ± 標本SD / 中央値。"
              f"成功は丸める前の速度MLP R² > 0.8。分類は各フレーム中心輝度の平均 ≥ {THRESHOLD}。"
              f"smooth_l1 の β = σ·√(π/2)。既存同一設定の列はステージ1〜3の同一設定の記録。\n\n")
    (out / "tables.md").write_text(header + markdown, encoding="utf-8")
    _write_csv(out / "conditions.csv", table)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--plan", action="store_true", help="show counts only")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    runtime = runtime_info(resolve_device(args.device))
    cases = plan()
    rows = read_results()
    done = sum(digest(experiment_spec(c[-1], runtime)) in rows for c in cases)
    print(f"M: {len(cases)} configurations, {done} complete, {len(cases) - done} remaining "
          f"on {runtime['device']}", flush=True)
    if args.plan:
        return
    if not args.export_only:
        start = time.perf_counter()
        for i, (*_, cfg) in enumerate(cases, 1):
            run_one(cfg, runtime)
            if i % 10 == 0:
                print(f"M progress {i}/{len(cases)} after {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    print(f"Exported {export(read_results(), runtime)}/{len(cases)} M rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
