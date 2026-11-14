"""Summarize the replacement stage-3 design without mixing contrast, seed sets or noise placement."""
from collections import defaultdict
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from collapse_replay import LEDGER
from plot import with_reconstruction_metrics
from run_sweep import FILTERED_PROTOCOL, LEVELS, RADII, SEEDS, digest, identity, read_results, stage3_arms
from untrained_encoder import experiment_spec, PROTOCOL as UNTRAINED_PROTOCOL

TASKS = ("position", "velocity", "future")
PROBES = tuple(f"{task}_r2_mlp" for task in TASKS)
IMAGE = ("psnr", "corrected_psnr", "center_intensity")
METRICS = (*IMAGE, *PROBES)
DIMS = ("arm", "radius", "foreground", "noise_type", "noise_level", "placement", "loss")
THRESHOLD = .8


def statistics(values, seeds, probe):
    """Success uses strict, unrounded R²; seed IDs refer to the supplied order."""
    if not values:
        return None
    result = {"mean": mean(values), "median": median(values), "sd": stdev(values) if len(values) > 1 else None}
    if probe:
        passed = [seed for seed, v in zip(seeds, values, strict=True) if v > THRESHOLD]
        failed = [seed for seed, v in zip(seeds, values, strict=True) if v <= THRESHOLD]
        result.update(successes=len(passed), success_rate=len(passed) / len(seeds),
                      successful_seeds=passed, unsuccessful_seeds=failed)
    return result


def _metrics(row, names):
    if row["probe_splits"] != [6, 7]:
        raise ValueError(f"Unexpected probe holdout for {row['config_hash']}")
    result = {k: row["metrics"][k] for k in names}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError(f"Non-finite held-out result for {row['config_hash']}")
    return result


def collect(rows, observations, runtime, base, rates):
    if runtime["protocol"] != FILTERED_PROTOCOL:
        raise ValueError("The replacement stage 3 requires the filtered final protocol")
    arms = stage3_arms(base, rates)
    d_refs = list(dict.fromkeys(replace(c, placement="target_only") for c in arms["D"]))
    untrained_runtime = {**runtime, "protocol": UNTRAINED_PROTOCOL}
    baselines = {}
    for r in RADII:
        for seed in SEEDS:
            config = replace(base, radius=r, seed=seed)
            key = digest(experiment_spec(config, untrained_runtime))
            if key in rows:
                row = rows[key]
                baselines[r, seed] = {"radius": r, "seed": seed, "config_hash": key,
                                      "model_sha256": row["model_sha256"], **_metrics(row, PROBES)}

    grouped = defaultdict(list)
    for arm, cases in arms.items():
        for cfg in cases + (d_refs if arm == "D" else []):
            key = (arm, cfg.radius, cfg.foreground, cfg.noise_type, cfg.noise_level, cfg.placement, cfg.loss)
            grouped[key].append(cfg)

    conditions, per_seed = [], []
    for key, cases in grouped.items():
        ids = sorted(c.seed for c in cases)
        if len(set(ids)) != len(ids):
            raise ValueError(f"Repeated seed in stage-3 group {key}")
        cell = dict(zip(DIMS, key, strict=True))
        observed = []
        for c in sorted(cases, key=lambda x: x.seed):
            cfg_hash = digest(identity(c, "experiment", runtime))
            if cfg_hash not in rows:
                continue
            row = rows[cfg_hash]
            if row["first_stage"] not in (1, 2, 3):
                raise ValueError(f"Unexpected source stage for {cfg_hash}")
            joined = with_reconstruction_metrics(row, observations)
            metrics = _metrics({**row, "metrics": joined}, METRICS)
            measured = {**cell, "seed": c.seed, "config_hash": cfg_hash, "model_sha256": row["model_sha256"],
                        "first_stage": row["first_stage"], **metrics}
            baseline = baselines.get((c.radius, c.seed))
            for name in PROBES:
                measured["untrained_" + name] = baseline[name] if baseline is not None else None
            observed.append(measured)
            per_seed.append(measured)
        seeds_done = [x["seed"] for x in observed]
        baseline_ids = [s for s in ids if (cell["radius"], s) in baselines]
        conditions.append({**cell, "expected_seeds": ids, "observed_seeds": seeds_done,
                           "complete": seeds_done == ids,
                           "metrics": {k: statistics([x[k] for x in observed], seeds_done, k in PROBES)
                                       for k in METRICS},
                           "untrained_seeds": baseline_ids,
                           "untrained": {k: statistics([baselines[cell["radius"], s][k] for s in baseline_ids],
                                                       baseline_ids, True) for k in PROBES}})
    unique = {x["config_hash"]: x for x in per_seed}
    design = set(c for cases in arms.values() for c in cases)
    design_keys = {digest(identity(c, "experiment", runtime)) for c in design}
    return {"runtime": runtime, "frame_gap": base.frame_gap, "steps": base.steps, "latent_dim": base.latent_dim,
            "learning_rates": {k: rates[k] for k in ("l1", "l2")}, "background": base.background,
            "success_rule": "R² > 0.8", "probe_splits": [6, 7],
            "expected_stage3_design": len(design), "observed_stage3_design": len(design_keys & unique.keys()),
            "expected_unique_experiments": len(design | set(d_refs)),
            "observed_unique_experiments": len(unique),
            "observed_first_stage_3": sum(r["first_stage"] == 3 for r in unique.values()),
            "conditions": conditions, "seeds": per_seed, "untrained": list(baselines.values())}


def mixed_seeds(summary):
    """Return complete conditions straddling .8, retaining the exact per-seed values."""
    scores = defaultdict(dict)
    for record in summary["seeds"]:
        scores[tuple(record[k] for k in DIMS)][record["seed"]] = record
    mixed = []
    for cell in summary["conditions"]:
        if not cell["complete"]:
            continue
        dims = {k: cell[k] for k in DIMS}
        for name in PROBES:
            stats = cell["metrics"][name]
            if 0 < stats["success_rate"] < 1:
                vals = scores[tuple(dims.values())]
                mixed.append({**dims, "metric": name, "successful_seeds": stats["successful_seeds"],
                              "unsuccessful_seeds": stats["unsuccessful_seeds"],
                              "scores": {s: vals[s][name] for s in cell["observed_seeds"]}})
    return mixed


def _csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                             for k, v in r.items()})


def condition_rows(summary):
    for cell in summary["conditions"]:
        row = {k: cell[k] for k in DIMS}
        row.update(expected_seeds=cell["expected_seeds"], observed_seeds=cell["observed_seeds"],
                   complete=cell["complete"], untrained_seeds=cell["untrained_seeds"])
        for prefix, values in (("", cell["metrics"]), ("untrained_", cell["untrained"])):
            for metric, stats in values.items():
                fields = ("mean", "median", "sd")
                if metric in PROBES:
                    fields += ("successes", "success_rate")
                for stat in fields:
                    row[f"{prefix}{metric}_{stat}"] = stats[stat] if stats is not None else None
        yield row


def draw_heatmaps(summary, out):
    data = {(x["radius"], x["noise_level"], x["loss"]):x for x in summary["conditions"] if x["arm"] == "A"}
    specs = [("velocity_r2_mlp", "success_rate", "Velocity MLP success rate", "velocity_success", "R² > 0.8"),
             ("center_intensity", "mean", "Mean reconstructed center brightness", "center_intensity", "Intensity"),
             ("corrected_psnr", "mean", "Mean affine-corrected PSNR", "corrected_psnr", "dB")]
    levels = LEVELS["gaussian"]
    for metric, stat, title, stem, units in specs:
        panels = []
        for loss in ("l1", "l2"):
            panel = np.full((len(RADII), len(levels)), np.nan)
            for i, radius in enumerate(RADII):
                for j, sigma in enumerate(levels):
                    cell = data[radius, sigma, loss]
                    if cell["complete"]:
                        panel[i, j] = cell["metrics"][metric][stat]
            panels.append(panel)
        observed = np.concatenate([p[np.isfinite(p)] for p in panels])
        if stat == "success_rate" or not observed.size:
            scale = Normalize(0., 1.)
        else:
            lo, hi = float(observed.min()), float(observed.max())
            if lo == hi:
                lo -= .01; hi += .01
            scale = Normalize(lo, hi)
        fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.25), layout="constrained")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#e4e6e5")
        for ax, panel, loss in zip(axes, panels, ("L1", "L2"), strict=True):
            im = ax.imshow(panel, cmap=cmap, norm=scale, interpolation="nearest", aspect="auto")
            ax.set_xticks(range(len(levels)), [f"{s:g}" for s in levels])
            ax.set_yticks(range(len(RADII)), RADII)
            ax.set(xlabel="Gaussian σ in target", ylabel="Radius (px)", title=loss)
            for i, j in np.ndindex(panel.shape):
                val = panel[i,j]
                cell = data[RADII[i], levels[j], loss.lower()]
                if not math.isfinite(val):
                    ax.text(j, i, f"{len(cell['observed_seeds'])}/5 done", ha="center", va="center", fontsize=8)
                    continue
                text = f"{cell['metrics'][metric]['successes']}/5" if stat == "success_rate" else f"{val:.2f}"
                rr, gg, bb, _ = cmap(scale(val))
                color = "black" if .299*rr + .587*gg + .114*bb > .55 else "white"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
        fig.suptitle(f"{title}\nFull contrast · 5 training seeds per cell", fontsize=13)
        fig.colorbar(im, ax=axes, shrink=.84, label=units)
        fig.savefig(out / f"arm_a_{stem}.png", dpi=180)
        plt.close(fig)


def _task_cell(cell, name):
    s = cell["metrics"][name]
    if s is None:
        return "未実行"
    return f"{s['mean']:.3f} / {s['median']:.3f} / {s['successes']}/{len(cell['observed_seeds'])}"


def report_tables(summary):
    """Produce Japanese Markdown tables from the same seed memberships used for plotting."""
    sections = {"A":"全半径・全ガウス量", "B":"全半径・塩胡椒2水準", "C":"低コントラスト・ガウス",
                "D":"ノイズ位置の比較"}
    parts = ["各R²欄は順にシード平均、中央値、R² > 0.8 の成功数 / 評価シード数。"
             "PSNRと中心輝度はシード平均。中心輝度は両フレームの真の中心位置で補間した絶対輝度。"
             "補正PSNRは各モデルについて評価5,000組の全画素で1組の係数を求めた値。"]
    for arm, title in sections.items():
        parts += ["", f"### {arm}　{title}", "",
                  "| r | 物体輝度 | ノイズ | 量 | ノイズの場所 | 損失 | 通常PSNR | 補正PSNR | 中心輝度 | 位置R² | 速度R² | 未来位置R² |",
                  "|---:|---:|---|---:|---|---|---:|---:|---:|---|---|---|"]
        for cell in summary["conditions"]:
            if cell["arm"] != arm:
                continue
            cols = [str(cell["radius"]), f"{cell['foreground']:g}", cell["noise_type"], f"{cell['noise_level']:g}",
                    cell["placement"], cell["loss"].upper()]
            for key, decimals in zip(IMAGE, (2,2,4), strict=True):
                s = cell["metrics"][key]
                cols.append(f"{s['mean']:.{decimals}f}" if s is not None else "未実行")
            cols.extend(_task_cell(cell, name) for name in PROBES)
            parts.append("| " + " | ".join(cols) + " |")
    parts += ["", "### 半径別の未学習エンコーダ", "",
              "各半径のクリーン画像、物体輝度1.0で測った初期化直後の5シードを再掲した。"
              "Cの低コントラストとDの入力側ノイズは、この未学習測定の入力条件と異なる。"
              "条件別CSVにはBを含め、各比較と同じシード番号に絞った基準値も記録した。",
              "", "| r | 位置R² | 速度R² | 未来位置R² |", "|---:|---|---|---|"]
    for rad in RADII:
        z = sorted((x for x in summary["untrained"] if x["radius"] == rad), key=lambda x:x["seed"])
        seeds = [x["seed"] for x in z]
        m = {k: statistics([x[k] for x in z], seeds, True) for k in PROBES}
        parts.append(f"| {rad} | " + " | ".join(_task_cell({"metrics":m,"observed_seeds":seeds}, k) for k in PROBES) + " |")
    parts += ["", "### 同じ条件で成功・未達の両方が出たシード", "",
              "全予定シードを採点し、成功率が0と1の間になった条件を掲載する。数値は丸める前のR²で判定した。", ""]
    m = mixed_seeds(summary)
    if m:
        parts += ["| 実験 | r | 輝度 | ノイズ量・位置 | 損失 | 指標 | 成功seed | 未達seed | seed=R² |",
                  "|---|---:|---:|---|---|---|---|---|---|"]
        for x in m:
            label = dict(zip(PROBES, ("位置", "速度", "未来位置"), strict=True))[x["metric"]]
            scores = ", ".join(f"{s}={v:.4f}" for s,v in x["scores"].items())
            parts.append(f"| {x['arm']} | {x['radius']} | {x['foreground']:g} | {x['noise_type']} {x['noise_level']:g} "
                         f"{x['placement']} | {x['loss'].upper()} | {label} | {x['successful_seeds']} | "
                         f"{x['unsuccessful_seeds']} | {scores} |")
    else:
        parts.append("完了した条件の該当例は0件。")
    return "\n".join(parts)


def write_outputs(rows, runtime, base, rates, out):
    summary = collect(rows, read_results(LEDGER), runtime, base, rates)
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    (out / "stage3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    seed_fields = (*DIMS, "seed", "config_hash", "model_sha256", "first_stage", *METRICS, *("untrained_"+p for p in PROBES))
    _csv(out / "stage3_seeds.csv", seed_fields, summary["seeds"])
    cells = list(condition_rows(summary))
    _csv(out / "stage3_conditions.csv", cells[0].keys(), cells)
    mixed = mixed_seeds(summary)
    _csv(out / "stage3_mixed_seeds.csv", (*DIMS,"metric","successful_seeds","unsuccessful_seeds","scores"), mixed)
    draw_heatmaps(summary, out)
    print(f"Stage 3: {summary['observed_stage3_design']}/{summary['expected_stage3_design']} design fits, "
          f"{summary['observed_unique_experiments']}/{summary['expected_unique_experiments']} unique fits "
          f"including matched D controls. {out}")
    return summary
