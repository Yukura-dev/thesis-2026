"""K: collect the two initialization interventions and the unchanged historical controls."""
import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path

from final_summary import BASE, digest, load_ledger, statistics
from plot import with_reconstruction_metrics
from run_sweep import FILTERED_PROTOCOL, ROOT, RUNS

PURPOSE = "final_k_initialization"
TRAJECTORY_PURPOSE = "final_k_initialization_trajectory"
PROTOCOL = "circle-ae-final-k-initialization-v1"
CLASSES = ("background_only", "past_only", "current_only", "both")
CLASS_LABELS = dict(zip(CLASSES, ("背景のみ", "過去のみ", "現在のみ", "両方")))
VARIANTS = ("default", "a", "b")
VARIANT_LABELS = {"default": "既存", "a": "(a) 最終重みの縮小なし", "b": "(b) 既定初期化・背景加減算なし"}
CENTERS = ("center_intensity_previous", "center_intensity_current")
PROBES = tuple(f"{task}_r2_mlp" for task in ("position", "velocity", "future"))
METRICS = ("psnr", "corrected_psnr", *CENTERS, *PROBES)
TRAJECTORY_METRICS = ("output_mean", "background_mae", *CENTERS)
STEPS = (0, 300, 1000, 3000)
OUT = ROOT / "results/final_jkl/k"


def expected_config(radius, level, loss, seed):
    return {**BASE, "radius": radius, "noise_level": level, "loss": loss, "seed": seed}


def classify(previous, current):
    if not all(math.isfinite(v) for v in (previous, current)):
        raise ValueError("Classification needs finite measured centers")
    return CLASSES[int(previous >= .65) + 2 * int(current >= .65)]


def key_of(row, variant=None):
    cfg = row["config"]
    return (cfg["radius"], cfg["noise_level"], cfg["loss"], cfg["seed"],
            variant or row["runtime"]["initialization"])


def identify(rows, *, baseline=False):
    result = {}
    for row in rows.values():
        if baseline:
            if row["purpose"] != "experiment" or row["runtime"]["protocol"] != FILTERED_PROTOCOL:
                continue
            c = row["config"]
            if c != expected_config(c["radius"], c["noise_level"], c["loss"], c["seed"]):
                continue
            variant = "default"
        else:
            if row["purpose"] != PURPOSE:
                continue
            if row["runtime"]["protocol"] != PROTOCOL:
                raise ValueError("Unknown K protocol")
            c = row["config"]
            if c != expected_config(c["radius"], c["noise_level"], c["loss"], c["seed"]):
                raise ValueError("K changed a controlled setting")
            variant = row["runtime"]["initialization"]
            if variant not in ("a", "b"):
                raise ValueError("Unexpected initialization variant")
        if c["radius"] not in (5, 8) or c["noise_level"] not in (0., .4) or c["loss"] not in ("l1", "l2") or c["seed"] not in range(5):
            if baseline:
                continue
            raise ValueError("Unexpected K condition")
        key = key_of(row, variant)
        if key in result:
            raise ValueError(f"Duplicate condition or mixed runtime: {key}")
        if row.get("probe_splits") != [6, 7]:
            raise ValueError("K requires the unchanged probe splits 6/7")
        result[key] = row
    environments = {tuple(r["runtime"].get(k) for k in ("device", "numpy", "torch")) for r in result.values()}
    if len(environments) > 1:
        raise ValueError("Cannot pool different evaluation environments")
    return result


def collect(rows, frame_rows, reconstruction_rows, trajectory_rows, failures, *, allow_partial=False):
    source = {**identify(rows, baseline=True), **identify(rows)}
    frames = {}
    for r in frame_rows.values():
        source_key = r["source_config_hash"]
        if source_key in frames:
            raise ValueError("Multiple frame measurements for one model")
        frames[source_key] = r
    observed, pending = [], []
    failure_index = {key_of(r): r for r in failures.values()}
    for radius, level, loss, variant, seed in product((5, 8), (0., .4), ("l1", "l2"), VARIANTS, range(5)):
        key = radius, level, loss, seed, variant
        row = source.get(key)
        if row is None or (variant == "default" and row["config_hash"] not in frames):
            pending.append(dict(radius=radius, noise_level=level, loss=loss, seed=seed,
                                initialization=variant, status="failed" if key in failure_index else "未実行"))
            continue
        metrics = dict(row["metrics"])
        frame_hash = None
        baseline = source.get((radius, level, loss, seed, "default"))
        if variant != "default" and baseline is not None and row.get("baseline_config_hash") != baseline["config_hash"]:
            raise ValueError("K names a different historical baseline")
        if variant == "default":
            measured = frames[row["config_hash"]]
            if measured["model_sha256"] != row["model_sha256"]:
                raise ValueError("Historical frame measurements use different weights")
            metrics = with_reconstruction_metrics(row, reconstruction_rows)
            metrics = {**metrics, **{k: measured["metrics"][k] for k in CENTERS}}
            frame_hash = measured["config_hash"]
        if any(not math.isfinite(metrics[k]) for k in METRICS):
            raise ValueError("Non-finite completed result")
        if metrics["corrected_psnr"] + 1e-6 < metrics["psnr"]:
            raise ValueError("Affine fit worsened MSE")
        category = classify(*(metrics[k] for k in CENTERS))
        if variant != "default" and row.get("frame_classification") != category:
            raise ValueError("Producer and collector disagree about frame class")
        observed.append(dict(radius=radius, noise_level=level, loss=loss, seed=seed,
                             initialization=variant, frame_classification=category,
                             config_hash=row["config_hash"], model_sha256=row["model_sha256"],
                             frame_measurement_hash=frame_hash, **{k: metrics[k] for k in METRICS}))
    conditions = []
    for radius, level, loss, variant in product((5, 8), (0., .4), ("l1", "l2"), VARIANTS):
        group = [r for r in observed if (r["radius"], r["noise_level"], r["loss"], r["initialization"]) == (radius, level, loss, variant)]
        missing = [r for r in pending if (r["radius"], r["noise_level"], r["loss"], r["initialization"]) == (radius, level, loss, variant)]
        seeds = [r["seed"] for r in group]
        conditions.append(dict(radius=radius, noise_level=level, loss=loss, initialization=variant,
                               observed_seeds=seeds, pending=missing, complete=len(group) == 5,
                               counts={c: sum(r["frame_classification"] == c for r in group) for c in CLASSES},
                               metrics={m: statistics([r[m] for r in group], seeds, probe=m in PROBES) for m in METRICS}))
    trajectories = []
    seen = set()
    planned = set(product((5, 8), (0., .4), ("l1", "l2"), range(5), ("a", "b")))
    for row in trajectory_rows.values():
        if row["purpose"] != TRAJECTORY_PURPOSE:
            raise ValueError("Unknown trajectory measurement")
        key = key_of(row)
        step = row["step"]
        if key not in planned or step not in STEPS or (*key, step) in seen:
            raise ValueError("Unexpected or repeated trajectory point")
        if row["config"] != expected_config(*key[:4]):
            raise ValueError("Trajectory changed a controlled setting")
        rt = row["runtime"]
        endpoint_runtime = {k: v for k, v in rt.items() if k not in ("checkpoint_step", "endpoint_config_hash")}
        endpoint_hash = digest(dict(config=row["config"], purpose=PURPOSE, runtime=endpoint_runtime))
        if (rt.get("protocol") != PROTOCOL or rt.get("checkpoint_step") != step or
                rt.get("endpoint_config_hash") != endpoint_hash or row.get("endpoint_config_hash") != endpoint_hash or
                row.get("evaluation_seed_sequence") != [7, 0] or row.get("evaluation_n") != 5000):
            raise ValueError("Trajectory protocol, endpoint identity or evaluation split differs")
        seen.add((*key, step))
        if any(not math.isfinite(row["metrics"][k]) for k in TRAJECTORY_METRICS):
            raise ValueError("Non-finite trajectory point")
        final = source.get(key)
        if final and final["config_hash"] != endpoint_hash:
            raise ValueError("Trajectory refers to a different endpoint")
        if final and step == 3000:
            if row["model_sha256"] != final["model_sha256"]:
                raise ValueError("Trajectory and endpoint weights disagree")
            for m in (*CENTERS, "background_mae"):
                if row["metrics"][m] != final["metrics"][m]:
                    raise ValueError("Trajectory and endpoint scores disagree")
        trajectories.append(dict(radius=key[0], noise_level=key[1], loss=key[2], seed=key[3],
                                 initialization=key[4], step=step, config_hash=row["config_hash"],
                                 model_sha256=row["model_sha256"],
                                 **{k: row["metrics"][k] for k in TRAJECTORY_METRICS}))
    trajectory_conditions = []
    for radius, level, loss, variant, step in product((5, 8), (0., .4), ("l1", "l2"), ("a", "b"), STEPS):
        group = sorted((r for r in trajectories if (r["radius"], r["noise_level"], r["loss"], r["initialization"], r["step"]) == (radius, level, loss, variant, step)), key=lambda r: r["seed"])
        seeds = [r["seed"] for r in group]
        trajectory_conditions.append(dict(radius=radius, noise_level=level, loss=loss, initialization=variant, step=step,
                                          observed_seeds=seeds,
                                          metrics={m: statistics([r[m] for r in group], seeds) for m in TRAJECTORY_METRICS}))
    complete = len(observed) == 120 and seen == {(*key, step) for key in planned for step in STEPS}
    if not complete and not allow_partial:
        raise ValueError(f"K is incomplete: {len(observed)}/120 endpoints, {len(trajectories)}/320 trajectory points; use --allow-partial to report failures or missing items explicitly")
    return dict(complete=complete, expected_new_fits=80, expected_controls=40, expected_trajectory_points=320,
                seeds=observed, conditions=conditions, trajectories=trajectories,
                trajectory_conditions=trajectory_conditions, pending=pending, failures=list(failures.values()))


def number(stats, probe=False):
    if stats is None:
        return "未実行"
    sd = "—" if stats["sd"] is None else f"{stats['sd']:.3f}"
    s = f"{stats['mean']:.3f} ± {sd} / {stats['median']:.3f}"
    return s + (f" / {stats['successes']}/{stats['n']}" if probe else "")


def markdown(summary, prefix="results/final_jkl/k/"):
    lines = ["### K　初期化の比較", "", "各指標は平均 ± 標本SD / 中央値。MLP R²はさらに成功数/評価済みシード数（丸める前のR² > 0.8）を示す。四分類は過去・現在の中心輝度の平均を各々0.65と比較した。", "",
             f"[全シード]({prefix}seeds.csv) · [推移の全シード]({prefix}trajectory_seeds.csv)", "",
             "| r | Gaussian σ | 損失 | 初期化 | 背景のみ / 過去のみ / 現在のみ / 両方 | PSNR | 補正PSNR | 過去中心 | 現在中心 | 位置R² | 速度R² | 未来位置R² |",
             "|---:|---:|---|---|---|---|---|---|---|---|---|---|"]
    for c in summary["conditions"]:
        counts = " / ".join(str(c["counts"][k]) for k in CLASSES)
        vals = [str(c["radius"]), f"{c['noise_level']:g}", c["loss"], VARIANT_LABELS[c["initialization"]], counts]
        vals += [number(c["metrics"][m], m in PROBES) for m in METRICS]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "### K　学習前と学習中の再構成", "",
              "学習に使わない同じ5,000評価例を、0 / 300 / 1,000 / 3,000ステップで測定。通常の画像輝度を使い、クリップしない。初期の平均出力輝度も0ステップ欄に示す。既存初期化の300・1,000ステップの推移は今回未実行で、上の表では保存済み3,000ステップ終点を対照にした。", "",
              "| r | σ | 損失 | 初期化 | step | 出力平均輝度 | 背景MAE | 過去中心 | 現在中心 |",
              "|---:|---:|---|---|---:|---|---|---|---|"]
    for c in summary["trajectory_conditions"]:
        vals = [str(c["radius"]), f"{c['noise_level']:g}", c["loss"], c["initialization"], str(c["step"])]
        vals += [number(c["metrics"][m]) for m in TRAJECTORY_METRICS]
        lines.append("| " + " | ".join(vals) + " |")
    if summary["pending"]:
        lines += ["", "未完了項目（数値を補完していない）：", "", "```json", json.dumps(summary["pending"], ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def flat_conditions(conditions):
    for c in conditions:
        row = {k: v for k, v in c.items() if k not in ("metrics", "counts")}
        row.update({"count_" + k: v for k, v in c.get("counts", {}).items()})
        for metric, stats in c["metrics"].items():
            if stats is not None:
                row.update({f"{metric}_{stat}": value for stat, value in stats.items()})
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = collect(load_ledger(RUNS), load_ledger(ROOT / "results/frame_classification_measurements.jsonl"),
                      load_ledger(ROOT / "results/collapse_reconstruction.jsonl"),
                      load_ledger(ROOT / "results/initialization_trajectory.jsonl"),
                      load_ledger(ROOT / "results/initialization_failures.jsonl"), allow_partial=args.allow_partial)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for filename, rows in (("seeds.csv", summary["seeds"]), ("conditions.csv", list(flat_conditions(summary["conditions"]))),
                           ("trajectory_seeds.csv", summary["trajectories"]), ("trajectory_conditions.csv", list(flat_conditions(summary["trajectory_conditions"])))):
        csv_write(args.out / filename, rows)
    (args.out / "tables.md").write_text(markdown(summary, prefix=""))
    print(f"K: {len(summary['seeds'])}/120 endpoints incl.40 controls; {len(summary['trajectories'])}/320 trajectory points")


if __name__ == "__main__":
    main()
