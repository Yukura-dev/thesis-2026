"""Aggregate the final loss-switch (G) and background-texture (H) experiments.

This module only reads completed results. --rescore-h-zero measures saved
stage-1 weights without any optimizer, so the untextured reference has the same
per-frame center fields as the new texture experiments.
"""
import argparse
import csv
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results/runs.jsonl"
ZERO_IMAGES = ROOT / "results/final_h_zero_images.jsonl"
ARCHIVAL = ROOT / "results/collapse_reconstruction.jsonl"
ZERO_PROTOCOL = "circle-ae-final-h-zero-images-v1"
G_PROTOCOL = "circle-ae-final-g-switch-v1"
H_PROTOCOL = "circle-ae-background-texture-v1"
MAIN_PROTOCOL = "circle-ae-filtered-final-v1"
METHODS = ("l2_6000", "l2_3000_l1_3000", "l1_6000")
METHOD_LABELS = dict(zip(METHODS, ("(i) L2 6,000", "(ii) L2 3,000 → L1 3,000", "(iii) L1 6,000"), strict=True))
G_LEVELS = {5: (0., .1, .2, .3, .45), 8: (0., .2, .45)}
TEXTURES = (0., .1, .4)
SEEDS = tuple(range(5))
TASKS = ("position", "velocity", "future")
PROBES = tuple(f"{task}_r2_{probe}" for probe in ("ridge", "mlp") for task in TASKS)
MLP = tuple(f"{task}_r2_mlp" for task in TASKS)
CENTERS = ("center_intensity_previous", "center_intensity_current")
METRICS = ("psnr", "corrected_psnr", "center_intensity", *CENTERS, *PROBES)
PAIRED = ("psnr", "corrected_psnr", "velocity_r2_mlp")
G_DIMS = ("radius", "noise_level", "method")
H_DIMS = ("radius", "background_texture_sd", "loss")
VISIBLE = .65
BASE = dict(background=.3, batch_size=128, foreground=1., frame_gap=4, latent_dim=16,
            loss="l2", lr=.003, noise_level=0., noise_type="gaussian", placement="target_only",
            probe_test=5000, probe_train=10000, radius=5, reject_reflections=True, seed=0, steps=3000)


def digest(spec):
    return sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_ledger(path):
    """Validate persisted identities and reject duplicate rows, even identical ones."""
    if not path.exists():
        return {}
    from filelock import FileLock
    with FileLock(str(path) + ".lock"):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {}
    for row in records:
        key = digest({name: row[name] for name in ("config", "purpose", "runtime")})
        if key != row["config_hash"] or key in indexed:
            raise ValueError(f"Invalid or duplicate result identity: {row['config_hash']}")
        indexed[key] = row
    return indexed


def one_runtime(rows, purpose, protocol, *, drop=()):
    possibilities = {}
    for row in rows.values():
        if row["purpose"] == purpose and row["runtime"]["protocol"] == protocol:
            runtime = {k: v for k, v in row["runtime"].items() if k not in drop}
            possibilities[digest(runtime)] = runtime
    if len(possibilities) > 1:
        raise ValueError(f"Multiple environments for {purpose}; do not mix them")
    return next(iter(possibilities.values()), None)


def g_config(radius, level, method, seed, *, anchor=False):
    return {**BASE, "radius": radius, "noise_level": level,
            "noise_type": "salt_pepper" if level else "gaussian", "seed": seed,
            "steps": 3000 if anchor else 6000,
            "loss": "l2" if anchor or method == "l2_6000" else "l1"}


def h_config(radius, texture, loss, seed):
    cfg = {**BASE, "radius": radius, "loss": loss, "seed": seed}
    if texture:
        cfg["background_texture_sd"] = texture
    return cfg


def zero_spec(source):
    return {"config": source["config"], "purpose": "final_h_zero_images",
            "runtime": {**source["runtime"], "protocol": ZERO_PROTOCOL,
                        "source_config_hash": source["config_hash"]}}


def statistics(values, seeds, *, probe=False):
    if len(values) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Every score must have one unique seed")
    if not values:
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite completed result")
    out = dict(n=len(values), mean=mean(values), sd=stdev(values) if len(values) > 1 else None,
               median=median(values))
    if probe:
        passed = [seed for seed, value in zip(seeds, values, strict=True) if value > .8]
        failed = [seed for seed, value in zip(seeds, values, strict=True) if value <= .8]
        out.update(successes=len(passed), success_rate=len(passed) / len(values),
                   successful_seeds=passed, unsuccessful_seeds=failed)
    return out


def checked_metrics(row):
    if row.get("probe_splits") != [6, 7]:
        raise ValueError(f"Unexpected held-out split: {row['config_hash']}")
    values = {name: row["metrics"][name] for name in METRICS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"Non-finite measurement: {row['config_hash']}")
    if abs(sum(values[name] for name in CENTERS) / 2 - values["center_intensity"]) > 1e-10:
        raise ValueError(f"Frame centers disagree with pooled center: {row['config_hash']}")
    if values["corrected_psnr"] + 1e-6 < values["psnr"]:
        raise ValueError(f"Affine correction worsened pooled MSE: {row['config_hash']}")
    if row.get("runtime", {}).get("protocol") in (G_PROTOCOL, H_PROTOCOL):
        if row["metrics"]["affine_n_pixels"] != row["config"]["probe_test"] * 2 * 64 * 64:
            raise ValueError("Affine correction must fit one transform over all held-out pixels")
    return values


def summarize(records, dims, cases, metrics=METRICS, *, classify_success=True):
    indexed = {tuple(row[k] for k in (*dims, "seed")): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate seed in summary")
    expected_keys = {(*case, seed) for case in cases for seed in SEEDS}
    if not indexed.keys() <= expected_keys:
        raise ValueError("An unplanned condition entered the summary")
    result = []
    for case in cases:
        seeds = [seed for seed in SEEDS if (*case, seed) in indexed]
        result.append({**dict(zip(dims, case, strict=True)), "expected_seeds": list(SEEDS),
                       "observed_seeds": seeds, "complete": seeds == list(SEEDS),
                       "metrics": {name: statistics([indexed[*case, s][name] for s in seeds], seeds,
                                                    probe=classify_success and name in PROBES) for name in metrics}})
    return result


def paired_comparison(records):
    indexed = {(row["radius"], row["noise_level"], row["method"], row["seed"]): row for row in records}
    paired = []
    for radius, levels in G_LEVELS.items():
        for level, seed in product(levels, SEEDS):
            control = indexed.get((radius, level, METHODS[0], seed))
            switched = indexed.get((radius, level, METHODS[1], seed))
            if control is None or switched is None:
                continue
            if control["anchor_config_hash"] != switched["anchor_config_hash"]:
                raise ValueError("Paired methods must begin with the same three-thousand-step L2 model")
            if control["data_stream_state_sha256"] != switched["data_stream_state_sha256"]:
                raise ValueError("Paired methods did not continue the same complete data stream")
            row = dict(radius=radius, noise_level=level, seed=seed,
                       control_config_hash=control["config_hash"], switched_config_hash=switched["config_hash"],
                       anchor_config_hash=control["anchor_config_hash"])
            for name in PAIRED:
                row[name + "_control"] = control[name]
                row[name + "_switched"] = switched[name]
                row[name] = switched[name] - control[name]
            paired.append(row)
    # A difference is not an R² score: no .8 success classification on differences.
    cells = summarize(paired, ("radius", "noise_level"),
                      [(r, p) for r, levels in G_LEVELS.items() for p in levels], PAIRED,
                      classify_success=False)
    return paired, cells


def switch_comparison(records, anchors):
    comparisons, disappeared = [], []
    for row in records:
        if row["method"] != "l2_3000_l1_3000":
            continue
        before = anchors.get(row["anchor_config_hash"])
        if before is None:
            raise ValueError("Switch endpoint has no own L2 anchor")
        if (row["radius"], row["noise_level"], row["seed"]) != tuple(before[k] for k in ("radius", "noise_level", "seed")):
            raise ValueError("Switch comparison mismatches radius, corruption or seed")
        change = {k: row[k] for k in ("radius", "noise_level", "seed", "config_hash", "model_sha256")}
        change.update(before_config_hash=before["config_hash"], before_model_sha256=before["model_sha256"],
                      center_threshold=VISIBLE)
        for name in METRICS:
            change["before_" + name] = before[name]
            change["after_" + name] = row[name]
        for frame, name in zip(("previous", "current"), CENTERS, strict=True):
            was = before[name] >= VISIBLE
            now = row[name] >= VISIBLE
            change["visible_before_" + frame] = was
            change["visible_after_" + frame] = now
            change["disappeared_" + frame] = was and not now
        comparisons.append(change)
        for frame in ("previous", "current"):
            if change["disappeared_" + frame]:
                disappeared.append({**change, "disappeared_frame": frame})
    return comparisons, disappeared


def collect_g(rows, runtime):
    records, anchor_rows = [], {}
    for radius, levels in G_LEVELS.items():
        for level, seed in product(levels, SEEDS):
            cfg = g_config(radius, level, "l2_6000", seed, anchor=True)
            anchor_key = digest(dict(config=cfg, purpose="final_g_l2_anchor", runtime=runtime)) if runtime else None
            anchor = rows.get(anchor_key)
            if anchor is not None:
                check_training_history(anchor, rows)
                anchor_rows[anchor_key] = dict(radius=radius, noise_level=level, seed=seed,
                                               config_hash=anchor_key, model_sha256=anchor["model_sha256"],
                                               **checked_metrics(anchor))
            for method in METHODS:
                cfg = g_config(radius, level, method, seed)
                rt = {**runtime, "method": method} if runtime else None
                key = digest(dict(config=cfg, purpose="final_g_switch", runtime=rt)) if rt else None
                row = rows.get(key)
                if row is None:
                    continue
                if row["method"] != method:
                    raise ValueError(f"Method metadata disagrees with result identity: {key}")
                check_training_history(row, rows)
                uses_anchor = method != "l1_6000"
                if uses_anchor and (row["anchor_config_hash"] != anchor_key or anchor is None or
                                    row["anchor_model_sha256"] != anchor["model_sha256"] or
                                    row["anchor_data_stream_state_sha256"] != anchor["data_stream_state_sha256"]):
                    raise ValueError(f"Switch/continuation endpoint has a mismatched anchor: {key}")
                records.append(dict(radius=radius, noise_level=level, method=method, seed=seed,
                                    config_hash=key, model_sha256=row["model_sha256"],
                                    anchor_config_hash=anchor_key if uses_anchor else None,
                                    anchor_model_sha256=anchor["model_sha256"] if uses_anchor else None,
                                    data_stream_state_sha256=row["data_stream_state_sha256"],
                                    **checked_metrics(row)))
    for radius, levels in G_LEVELS.items():
        for level, seed in product(levels, SEEDS):
            triplet = [r for r in records if (r["radius"], r["noise_level"], r["seed"]) == (radius, level, seed)]
            if len({r["data_stream_state_sha256"] for r in triplet}) > 1:
                raise ValueError("G methods consumed different motion/noise trajectories for the same seed")
            counts = [rows[r["config_hash"]]["train_window_counts"] for r in triplet]
            if counts and any(value != counts[0] for value in counts[1:]):
                raise ValueError("G methods consumed different numbers of proposed/accepted trajectories")
    cases = [(r, p, method) for r, levels in G_LEVELS.items() for p, method in product(levels, METHODS)]
    paired, paired_cells = paired_comparison(records)
    switched, gone = switch_comparison(records, anchor_rows)
    return dict(runtime=runtime, expected_endpoints=120, observed_endpoints=len(records),
                expected_anchors=40, observed_anchors=len(anchor_rows), seeds=records,
                anchors=list(anchor_rows.values()), conditions=summarize(records, G_DIMS, cases),
                paired_difference="l2_3000_l1_3000 minus l2_6000",
                paired_seeds=paired, paired_conditions=paired_cells,
                switch_comparisons=switched, disappeared_frames=gone,
                disappearance_rule="own 3000-step L2 frame center >= 0.65 and switched 6000-step frame center < 0.65")


def check_training_history(row, rows):
    """Reject unfair continuation/reset or reuse metadata rather than silently average it."""
    cfg, method = row["config"], row.get("method")
    switched = method == "l2_3000_l1_3000"
    phases = [[0, 3000, "l2"], [3000, 6000, "l1"]] if switched else [[0, cfg["steps"], cfg["loss"]]]
    if (row["training_phases"] != phases or row["optimizer_reset_at"] != (3000 if switched else None) or
            row["training_seed_sequence"] != [0, cfg["seed"]] or
            not row["continued_motion_and_noise_streams"] or
            row["train_window_counts"]["accepted"] != cfg["steps"] * cfg["batch_size"]):
        raise ValueError(f"Invalid G training history: {row['config_hash']}")
    for item in row["provenance"]:
        source = rows[item["source_config_hash"]]
        if source["model_sha256"] != item["model_sha256"]:
            raise ValueError("Provenance names weights that differ from the source model")
        if item["kind"] == "measured_probes_reused_images_verified":
            if row["model_sha256"] != source["model_sha256"]:
                raise ValueError("Scores were reused for a different endpoint model")
            for name in row["metrics"].keys() & source["metrics"].keys():
                if row["metrics"][name] != source["metrics"][name]:
                    raise ValueError(f"A reused measurement changed: {name}")


def zero_metrics(source, images, archival):
    measured = images[digest(zero_spec(source))]
    if (measured["source_config_hash"] != source["config_hash"] or
            measured["model_sha256"] != source["model_sha256"] or
            measured["probe_score_stream"] != [7, 0] or not measured["original_image_metrics_exact"] or
            not measured["weights_only"] or measured["evaluation_pairs"] != source["config"]["probe_test"]):
        raise ValueError("Untextured frame measurements do not match the original model")
    archive_rt = {**source["runtime"], "protocol": "circle-ae-collapse-image-v1",
                  "source_config_hash": source["config_hash"]}
    archive_key = digest(dict(config=source["config"], purpose="collapse_reconstruction", runtime=archive_rt))
    old = archival[archive_key]
    if (old["source_config_hash"] != source["config_hash"] or old["model_sha256"] != source["model_sha256"] or
            old["probe_score_stream"] != [7, 0] or not old["original_validation_exact"] or
            not old["original_reconstruction_metrics_exact"] or
            old["metrics"]["n_pixels"] != source["config"]["probe_test"] * 2 * 64 * 64):
        raise ValueError("Untextured affine measurements do not match the original model")
    for name in ("psnr", "object_mae", "background_mae", "center_intensity", "center_contrast"):
        if measured["metrics"][name] != source["metrics"][name] or old["metrics"][name] != source["metrics"][name]:
            raise ValueError("Untextured reference image scores differ from the archived experiment")
    merged = {**source["metrics"], **{name: measured["metrics"][name] for name in CENTERS},
              "corrected_psnr": old["metrics"]["corrected_psnr"]}
    return checked_metrics({**source, "metrics": merged}), measured["config_hash"], archive_key


def collect_h(rows, images, archival, runtime, main_runtime):
    records = []
    for radius, texture, loss, seed in product((5, 8), TEXTURES, ("l1", "l2"), SEEDS):
        cfg = h_config(radius, texture, loss, seed)
        rt, purpose = (runtime, "background_texture") if texture else (main_runtime, "experiment")
        key = digest(dict(config=cfg, purpose=purpose, runtime=rt)) if rt else None
        row = rows.get(key)
        if row is None:
            continue
        provenance = dict(frame_metrics_config_hash=key, affine_metrics_config_hash=key)
        if texture:
            if (row["evaluation_reference"] != "full textured clean image; PSNR fixed data range 1" or
                    row["runtime"]["texture_method"]["clipping"] != "none"):
                raise ValueError("H must use the authorized unclipped texture and textured PSNR reference")
            metrics = checked_metrics(row)
        else:
            if digest(zero_spec(row)) not in images:
                continue
            metrics, frame_key, affine_key = zero_metrics(row, images, archival)
            provenance = dict(frame_metrics_config_hash=frame_key, affine_metrics_config_hash=affine_key)
        records.append(dict(radius=radius, background_texture_sd=texture, loss=loss, seed=seed,
                            config_hash=key, model_sha256=row["model_sha256"], **provenance, **metrics))
    cases = list(product((5, 8), TEXTURES, ("l1", "l2")))
    return dict(runtime=runtime, zero_texture_runtime=main_runtime, expected_endpoints=60,
                observed_endpoints=len(records), seeds=records,
                conditions=summarize(records, H_DIMS, cases),
                psnr_reference="the clean image including the sample's fixed background texture",
                psnr_data_range=1., texture_sd_definition="population SD over the entire 64x64 background before circle overlay")


def collect(rows, images, archival, *, allow_partial=False):
    g_rt = one_runtime(rows, "final_g_switch", G_PROTOCOL, drop=("method",))
    anchor_rt = one_runtime(rows, "final_g_l2_anchor", G_PROTOCOL)
    if g_rt is None:
        g_rt = anchor_rt
    elif anchor_rt is not None and anchor_rt != g_rt:
        raise ValueError("G anchor and endpoint environments differ")
    h_rt = one_runtime(rows, "background_texture", H_PROTOCOL)
    main_rt = one_runtime(rows, "experiment", MAIN_PROTOCOL)
    summary = dict(success_rule="R² > 0.8 using unrounded values", sample_sd="ddof=1 across training seeds",
                   probe_splits=[6, 7], G=collect_g(rows, g_rt),
                   H=collect_h(rows, images, archival, h_rt, main_rt))
    complete = (summary["G"]["observed_endpoints"] == 120 and summary["G"]["observed_anchors"] == 40 and
                summary["H"]["observed_endpoints"] == 60)
    summary["complete"] = complete
    if not complete and not allow_partial:
        raise ValueError(f"Incomplete G/H: G={summary['G']['observed_endpoints']}/120, "
                         f"anchors={summary['G']['observed_anchors']}/40, H={summary['H']['observed_endpoints']}/60")
    return summary


def csv_write(path, records, headers):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def flat_conditions(cells):
    for cell in cells:
        record = {k: v for k, v in cell.items() if k != "metrics"}
        for name, stats in cell["metrics"].items():
            fields = ("n", "mean", "sd", "median")
            if stats is not None and "successes" in stats:
                fields += ("successes", "success_rate", "successful_seeds", "unsuccessful_seeds")
            elif name in PROBES:
                # Paired differences intentionally do not receive success fields.
                if any(other["metrics"][name] is not None and "successes" in other["metrics"][name] for other in cells):
                    fields += ("successes", "success_rate", "successful_seeds", "unsuccessful_seeds")
            for field in fields:
                record[name + "_" + field] = stats[field] if stats is not None else None
        yield record


def _number(stats, digits=3, *, probe=False):
    if stats is None:
        return "未実行"
    sd = f"{stats['sd']:.{digits}f}" if stats["sd"] is not None else "—"
    result = f"{stats['mean']:.{digits}f} ± {sd} / {stats['median']:.{digits}f}"
    if probe:
        result += f" / {stats['successes']}/{stats['n']}"
    return result


def markdown(summary):
    parts = ["集計値は5シードの平均 ± 標本SD / 中央値。R²欄にはさらに丸める前の値が0.8を超えた成功数/測定数を示す。"
             "位置・速度・未来位置はMLP。リッジを含む全シードの数値は"
             "[Gの120件](results/final_gh/g_seeds.csv)、[Hの60件](results/final_gh/h_seeds.csv)に保存した。", ""]
    for study, dims, labels in (("G", G_DIMS, ("r", "塩胡椒p", "学習方法")),
                                ("H", H_DIMS, ("r", "背景テクスチャSD", "損失"))):
        parts.extend([f"### {study} の結果", "", "| " + " | ".join((*labels, "PSNR", "補正PSNR", "過去の中心輝度", "現在の中心輝度", "位置R²", "速度R²", "未来位置R²")) + " |",
                      "|" + "---|" * 10])
        for cell in summary[study]["conditions"]:
            values = [f"{cell[name]:g}" if isinstance(cell[name], (int, float)) else cell[name] for name in dims]
            if study == "G":
                values[2] = METHOD_LABELS[cell["method"]]
            values += [_number(cell["metrics"][name], 2 if "psnr" in name else 3) for name in ("psnr", "corrected_psnr", *CENTERS)]
            values += [_number(cell["metrics"][name], probe=True) for name in MLP]
            parts.append("| " + " | ".join(values) + " |")
        if study == "H":
            parts.extend(["", "Hの通常PSNRの真値は、そのサンプルの固定背景テクスチャを含むクリーン画像。"
                          "SDは円を重ねる前の64×64背景全体の母標準偏差で、PSNRの基準レンジは1に固定した。"
                          "背景と合成後の画像をクリップせず、値域外の値も保持した。"
                          "SD=0の20件は既存ステージ1の同じモデルを参照し、追加学習していない。"])
        parts.append("")
    parts.extend(["### G の対応するシード間の差", "", "差の符号は (ii) L2→L1 − (i) L2継続。"
                  "同じseedの差を先に計算し、その5つの差の平均 ± 標本SD / 中央値を示す。"
                  "[対応する40組の各シード差](results/final_gh/g_paired_seeds.csv)も保存した。", "",
                  "| r | p | 通常PSNR差 | 補正PSNR差 | 速度MLP R²差 |", "|---:|---:|---|---|---|"])
    for cell in summary["G"]["paired_conditions"]:
        parts.append(f"| {cell['radius']} | {cell['noise_level']:g} | " +
                     " | ".join(_number(cell["metrics"][name]) for name in PAIRED) + " |")
    parts.extend(["", "### G の切替後に円の基準を下回ったフレーム", "", "同じseedの3,000ステップL2モデルの中心輝度が0.65以上で、"
                  "L1へ切り替えて計6,000ステップ後に0.65未満となったフレームのみを挙げる。"
                  "切替前から0.65未満だったフレームは消失に数えない。中心輝度は各5,000評価組で平均した値。", ""])
    vanished = summary["G"]["disappeared_frames"]
    if vanished:
        parts.extend(["| r | p | seed | フレーム | 過去中心 before→after | 現在中心 before→after | 位置R² before→after | 速度R² before→after | 未来位置R² before→after |",
                      "|---:|---:|---:|---|---|---|---|---|---|"])
        for row in vanished:
            vals = [str(row["radius"]), f"{row['noise_level']:g}", str(row["seed"]), row["disappeared_frame"]]
            vals += [f"{row['before_'+name]:.4f} → {row['after_'+name]:.4f}" for name in (*CENTERS, *MLP)]
            parts.append("| " + " | ".join(vals) + " |")
    else:
        parts.append("完了した切替比較で該当するフレームは0件。" if summary["complete"] else "未実行分を含むため、判定は未完了。")
    return "\n".join(parts) + "\n"


def write_outputs(summary, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_gh_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for label, dims in (("G", G_DIMS), ("H", H_DIMS)):
        rows = summary[label]["seeds"]
        fallback = (*dims, "seed", "config_hash", "model_sha256", *METRICS)
        csv_write(out / f"{label.lower()}_seeds.csv", rows, rows[0].keys() if rows else fallback)
        rows = list(flat_conditions(summary[label]["conditions"]))
        csv_write(out / f"{label.lower()}_conditions.csv", rows, list(dict.fromkeys(k for row in rows for k in row)))
    for field, filename, fallback in (
        ("anchors", "g_l2_anchors.csv", ("radius", "noise_level", "seed", "config_hash", *METRICS)),
        ("paired_seeds", "g_paired_seeds.csv", ("radius", "noise_level", "seed", *PAIRED)),
        ("switch_comparisons", "g_switch_frames.csv", ("radius", "noise_level", "seed", "config_hash")),
        ("disappeared_frames", "g_disappeared_frames.csv", ("radius", "noise_level", "seed", "disappeared_frame")),
    ):
        rows = summary["G"][field]
        csv_write(out / filename, rows, rows[0].keys() if rows else fallback)
    rows = list(flat_conditions(summary["G"]["paired_conditions"]))
    csv_write(out / "g_paired_conditions.csv", rows, list(dict.fromkeys(k for row in rows for k in row)))
    # The standalone table links to its sibling CSV files; REPORT uses root paths.
    (out / "tables.md").write_text(markdown(summary).replace("(results/final_gh/", "("))


def rescore_h_zero(device="auto"):
    """Only saved weights; exact original image metrics are required before recording."""
    from filelock import FileLock
    import os
    import torch
    from collapse_replay import CACHE, test_batches
    from diagnose_steps import build_model
    from evaluation import ImageMetrics
    from training import Config, _parameter_digest, resolve_device

    rows = load_ledger(RUNS)
    runtime = one_runtime(rows, "experiment", MAIN_PROTOCOL)
    device = resolve_device(device)
    if runtime is None or str(device) != runtime["device"]:
        raise ValueError("Historical image rescoring must use the original device")
    prior = load_ledger(ZERO_IMAGES)
    for radius, loss, seed in product((5, 8), ("l1", "l2"), SEEDS):
        cfg = h_config(radius, 0., loss, seed)
        source_key = digest(dict(config=cfg, runtime=runtime, purpose="experiment"))
        source = rows[source_key]
        spec = zero_spec(source)
        key = digest(spec)
        if key in prior:
            continue
        with FileLock(str(ROOT / "checkpoints/final_gh_gpu.lock")):
            checkpoint = torch.load(CACHE / source_key / "weights.pt", map_location="cpu", weights_only=True)
            config = Config(**cfg)
            model = build_model(config, device)
            model.load_state_dict(checkpoint["model"])
            if checkpoint["source_config_hash"] != source_key or _parameter_digest(model) != source["model_sha256"]:
                raise ValueError(f"Saved weight identity mismatch for {source_key}")
            model.eval()
            stats = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
            for batch, prediction in test_batches(model, config, device):
                stats.update(prediction, batch, radius)
            metrics = stats.result()
            old = {k: v for k, v in metrics.items() if k not in CENTERS}
            if any(source["metrics"][k] != v for k, v in old.items()):
                raise ArithmeticError(f"Historical image metrics are not exactly reproduced for {source_key}")
            row = {**spec, "config_hash": key, "source_config_hash": source_key,
                   "model_sha256": source["model_sha256"], "metrics": metrics,
                   "probe_score_stream": [7, 0], "evaluation_pairs": 5000,
                   "original_image_metrics_exact": True, "weights_only": True}
        with FileLock(str(ZERO_IMAGES) + ".lock"):
            existing = {value["config_hash"]: value for line in ZERO_IMAGES.read_text().splitlines()
                        if line.strip() for value in [json.loads(line)]} if ZERO_IMAGES.exists() else {}
            if key in existing:
                if (existing[key]["model_sha256"], existing[key]["metrics"]) != (row["model_sha256"], row["metrics"]):
                    raise ValueError("Conflicting image remeasurement of the same saved model")
            else:
                with ZERO_IMAGES.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        prior[key] = row
        print(f"H zero-texture frames verified: r={radius} {loss} seed={seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/final_gh")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--rescore-h-zero", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.rescore_h_zero:
        rescore_h_zero(args.device)
        return
    summary = collect(load_ledger(RUNS), load_ledger(ZERO_IMAGES), load_ledger(ARCHIVAL),
                      allow_partial=args.allow_partial)
    write_outputs(summary, args.out)
    print(f"G {summary['G']['observed_endpoints']}/120, H {summary['H']['observed_endpoints']}/60; {args.out}")


if __name__ == "__main__":
    main()
