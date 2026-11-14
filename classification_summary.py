"""J: summarize saved-model frame classifications without fitting any model.

The global population is unique by model/config identity. A model may belong to
multiple experiment arms, but it enters global and loss-group distributions once.
Old H classifications remain archival; this report uses the requested local
background contrast rule and retains the old raw-center comparison separately.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median

from final_summary import load_ledger
from frame_classification import (
    CLASSES, LEDGER, analysis_loss, classify_measurement, design, measurement_spec,
)
from run_sweep import ROOT, RUNS, digest

PROBES = tuple(f"{task}_r2_{kind}" for kind in ("ridge", "mlp")
               for task in ("position", "velocity", "future"))
RADII = (1, 2, 3, 5, 8)
LEVELS = (0., .05, .1, .2, .4)
ARMS = ("A", "B", "D", "G", "H")
EXPECTED_MEMBERSHIPS = {"A": 250, "B": 60, "D": 60, "G": 120, "H": 60}
CONDITION_DIMS = ("arm", "role", "radius", "noise_type", "noise_level", "placement",
                  "background_texture_sd", "method", "analysis_loss", "final_loss", "steps")
CLASS_JA = dict(zip(CLASSES, ("背景のみ", "過去のみ", "現在のみ", "両フレーム"), strict=True))
CLASS_SHORT = dict(zip(CLASSES, ("BG", "Past", "Now", "Both"), strict=True))
OUT = ROOT / "results" / "frame_classification"


def baseline_records(rows):
    result = []
    for source in rows.values():
        if (source["purpose"] != "untrained_encoder" or
                source["runtime"]["protocol"] != "circle-ae-untrained-encoder-v1"):
            continue
        cfg = source["config"]
        if not (cfg["frame_gap"] == 4 and cfg["reject_reflections"] and cfg["steps"] == 0
                and cfg["latent_dim"] == 16 and cfg["noise_level"] == 0
                and cfg["background"] == .3 and cfg["foreground"] == 1.
                and cfg.get("background_texture_sd", 0.) == 0.):
            continue
        if source["probe_splits"] != [6, 7] or cfg["probe_test"] != 5000:
            raise ValueError("Untrained baseline has a different evaluation population")
        result.append(dict(radius=cfg["radius"], seed=cfg["seed"],
                           source_config_hash=source["config_hash"], model_sha256=source["model_sha256"],
                           analysis_loss="not_applicable", classification="not_applicable",
                           **{name: source["metrics"][name] for name in PROBES}))
    if Counter((row["radius"], row["seed"]) for row in result) != Counter(
            (r, s) for r in RADII for s in range(5)):
        raise ValueError("Expected exactly the existing 25 radius-matched untrained baselines")
    return sorted(result, key=lambda row: (row["radius"], row["seed"]))


def collect(rows, measurements, *, device="mps", allow_partial=False):
    targets = design(rows, device)
    membership_counts = Counter(m["arm"] for t in targets for m in t["memberships"])
    if len(targets) != 514 or membership_counts != EXPECTED_MEMBERSHIPS:
        raise ValueError(f"Unexpected J population: {len(targets)}, {membership_counts}")
    records, missing = [], []
    for target in targets:
        source = target["source"]
        key = digest(measurement_spec(source, device))
        if key not in measurements:
            missing.append(dict(source_config_hash=source["config_hash"],
                                model_identity=target["model_identity"], memberships=target["memberships"]))
            continue
        measured = measurements[key]
        if (measured["source_config_hash"] != source["config_hash"] or
                measured["model_identity"] != target["model_identity"] or
                measured["model_sha256"] != source["model_sha256"] or
                not measured["original_image_metrics_exact"] or not measured["weights_only"] or
                measured["probe_score_stream"] != [7, 0] or measured["evaluation_pairs"] != 5000 or
                any(measured["original_image_metric_absolute_differences"].values())):
            raise ValueError(f"Saved-model provenance does not match: {key}")
        cfg, metrics = source["config"], measured["metrics"]
        classification = classify_measurement(metrics, cfg["foreground"], bool(cfg.get("background_texture_sd", 0.)))
        if any(measured[name] != value for name, value in classification.items()):
            raise ValueError(f"Stored classification differs from unrounded center means: {key}")
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"Non-finite image metric: {key}/{name}")
            if name in source["metrics"] and value != source["metrics"][name]:
                raise ValueError(f"Original image metric changed: {key}/{name}")
        record = dict(model_identity=target["model_identity"], source_config_hash=source["config_hash"],
                      source_aliases=target["source_aliases"], measurement_config_hash=key,
                      model_sha256=source["model_sha256"], memberships=target["memberships"],
                      **{name: cfg[name] for name in ("radius", "seed", "noise_type", "noise_level",
                                                    "placement", "steps", "foreground")},
                      background_texture_sd=cfg.get("background_texture_sd", 0.),
                      method=source.get("method", "standard"), analysis_loss=analysis_loss(source),
                      final_loss=cfg["loss"], **classification, **metrics,
                      **{name: source["metrics"][name] for name in PROBES})
        if any(not math.isfinite(record[name]) for name in PROBES):
            raise ValueError(f"Non-finite original probe: {key}")
        record["class_changed_from_raw_rule"] = record["classification"] != record["raw_comparison_classification"]
        records.append(record)
    if missing and not allow_partial:
        raise ValueError(f"未実行: {len(missing)} / 514 model measurements are missing")
    return records, baseline_records(rows), missing, targets


def unique_records(records):
    indexed = {}
    for row in records:
        key = row["model_identity"]
        if key in indexed:
            raise ValueError(f"Repeated identity in the global population: {key}")
        indexed[key] = row
    return list(indexed.values())


def expanded_memberships(records):
    return [{**row, **member} for row in records for member in row["memberships"]]


def condition_counts(records, targets):
    expected = defaultdict(list)
    for target in targets:
        source, cfg = target["source"], target["source"]["config"]
        base = {**cfg, "background_texture_sd": cfg.get("background_texture_sd", 0.)}
        base.update(method=source.get("method", "standard"), analysis_loss=analysis_loss(source), final_loss=cfg["loss"])
        for member in target["memberships"]:
            item = {**base, **member}
            expected[tuple(item[name] for name in CONDITION_DIMS)].append(cfg["seed"])
    measured = defaultdict(list)
    for row in expanded_memberships(records):
        measured[tuple(row[name] for name in CONDITION_DIMS)].append(row)
    if not measured.keys() <= expected.keys():
        raise ValueError("Unplanned membership in class count table")
    result = []
    for key, expected_seeds in sorted(expected.items()):
        group = measured[key]
        seeds = sorted(row["seed"] for row in group)
        if len(set(seeds)) != len(seeds) or len(set(expected_seeds)) != len(expected_seeds):
            raise ValueError("A seed is repeated within a condition")
        counts = Counter(row["classification"] for row in group)
        result.append({**dict(zip(CONDITION_DIMS, key, strict=True)),
                       "expected_seeds": sorted(expected_seeds), "observed_seeds": seeds,
                       "complete": seeds == sorted(expected_seeds), "n": len(group),
                       **{name: counts[name] for name in CLASSES}})
    return result


def probe_distribution(records, baselines):
    unique = unique_records(records)
    result = []
    for loss in ("all", "l1", "l2", "l2_to_l1"):
        for category in CLASSES:
            group = [row for row in unique if row["classification"] == category
                     and (loss == "all" or row["analysis_loss"] == loss)]
            result.append(distribution_row(group, population="trained_unique", analysis_loss=loss,
                                           classification=category, radius="all"))
    for radius in RADII:
        group = [row for row in baselines if row["radius"] == radius]
        result.append(distribution_row(group, population="untrained_reference", analysis_loss="not_applicable",
                                       classification="not_applicable", radius=radius))
    return result


def distribution_row(records, **identity):
    result = dict(**identity, n=len(records))
    for name in PROBES:
        scores = [row[name] for row in records]
        for statistic, fn in (("median", median), ("min", min), ("max", max)):
            result[name+"_"+statistic] = fn(scores) if scores else None
    return result


def modal_class(counts):
    maximum = max(counts.values(), default=0)
    return [name for name in CLASSES if maximum and counts.get(name, 0) == maximum]


def heatmap_data(cells):
    selected = {(row["analysis_loss"], row["radius"], row["noise_level"]): row
                for row in cells if row["arm"] == "A"}
    result = []
    for loss in ("l1", "l2"):
        for radius in RADII:
            for level in LEVELS:
                row = selected[loss, radius, level]
                counts = {name: row[name] for name in CLASSES}
                modes = modal_class(counts)
                maximum = max(counts.values())
                annotation = " / ".join(CLASS_SHORT[name] for name in modes) if modes else "Unrun"
                annotation += f"\n{maximum}/{row['n']}" + (" each" if len(modes) > 1 else "")
                result.append(dict(loss=loss, radius=radius, noise_level=level, n=row["n"],
                                   complete=row["complete"], counts=counts, modes=modes,
                                   tied=len(modes) > 1, annotation=annotation))
    return result


def save_heatmap(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Rectangle
    import numpy as np
    colors = ["#d8d8d8", "#68b7d6", "#f6bd60", "#83c595", "#f7f3ed", "white"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, 6.5, 1.), 6)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7), sharey=True)
    for ax, loss in zip(axes, ("l1", "l2"), strict=True):
        cells = {(r["radius"], r["noise_level"]): r for r in data if r["loss"] == loss}
        values = np.array([[CLASSES.index(cells[r, p]["modes"][0]) if len(cells[r, p]["modes"]) == 1
                            else 4 if cells[r, p]["modes"] else 5 for p in LEVELS] for r in RADII])
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for i, radius in enumerate(RADII):
            for j, level in enumerate(LEVELS):
                cell = cells[radius, level]
                if cell["tied"]:
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch="///", edgecolor="#a0a0a0", linewidth=0))
                ax.text(j, i, cell["annotation"], ha="center", va="center", fontsize=10,
                        bbox=dict(facecolor=colors[4], alpha=.85, edgecolor="none", pad=1) if cell["tied"] else None)
        ax.set(xticks=range(5), xticklabels=[f"{x:g}" for x in LEVELS], yticks=range(5),
               yticklabels=RADII, xlabel="Gaussian sigma (target only)", title=loss.upper())
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Object radius (px)")
    fig.suptitle("A: modal reconstructed-frame class across five seeds", fontsize=15)
    fig.legend(handles=[Patch(facecolor=colors[i], label=label) for i, label in enumerate(
        ("BG: neither center >= 0.65", "Past: previous frame only", "Now: current frame only", "Both: both frames"))]
        + [Patch(facecolor=colors[4], hatch="///", label="Tie: every modal class shown")],
        loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(bottom=.18, top=.88, wspace=.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def csv_write(path, rows):
    if not rows:
        path.write_text("status\n該当なし\n")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                         if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)


def markdown(summary):
    rows = [f"J：保存済み {summary['measured_models']} / 514 固有モデルを再評価。再学習なし。",
            "条件所属は重複を許すが、全体・損失別の分布はモデルと設定の同一性で重複を除外する。Gの40中間モデルは追加しない。",
            "平坦背景は各フレームの平均中心輝度 ≥ 0.65。Hのテクスチャ背景は平均（再構成中心 − 描画前背景中心）≥ 平均（物体輝度 − 描画前背景中心）/2。過去Hの判定は書き換えず、rawルールの比較列を残す。",
            "集計のクラスは各モデルの5,000評価例における中心平均で決まり、個々の評価例の可視性を意味しない。",
            "[モデル別](models.csv)・[条件所属別](memberships.csv)・[条件別4分類数](condition_counts.csv)・[プローブ分布](probe_distributions.csv)・[境界近傍](borderline.csv)・[Hの旧新判定](h_rule_comparison.csv)・[未学習25参照](untrained_reference.csv)。", "",
            "条件別の数は、背景のみ / 過去のみ / 現在のみ / 両フレームの順。", "",
            "| 群 | 役割 | r | ノイズ | 量 / 背景SD | 入れる場所 | 方法 | n / 予定 | 背景のみ | 過去のみ | 現在のみ | 両フレーム |",
            "|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["condition_counts"]:
        amount = row["background_texture_sd"] if row["arm"] == "H" else row["noise_level"]
        method = row["method"] if row["arm"] == "G" else row["analysis_loss"]
        noise = ("なし（背景模様）" if amount else "なし（平坦背景）") if row["arm"] == "H" else row["noise_type"]
        placement = "入力・教師で共通" if row["arm"] == "H" else row["placement"]
        amount_label = f"背景SD {amount:g}" if row["arm"] == "H" else f"{amount:g}"
        rows.append(f"| {row['arm']} | {row['role']} | {row['radius']} | {noise} | {amount_label} | {placement} | {method} | {row['n']} / {len(row['expected_seeds'])} | " + " | ".join(str(row[c]) for c in CLASSES) + " |")
    rows += ["", "プローブ値は中央値 [最小, 最大]。未学習参照は半径別5シードで、損失・再構成分類は適用しない。", "",
             "| 母集団 | 損失 | 再構成分類 | r | n | 位置MLP R² | 速度MLP R² | 未来MLP R² |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for row in summary["probe_distributions"]:
        cells = []
        for task in ("position", "velocity", "future"):
            prefix = f"{task}_r2_mlp_"
            cells.append("該当なし" if row["n"] == 0 else f"{row[prefix+'median']:.5f} [{row[prefix+'min']:.5f}, {row[prefix+'max']:.5f}]")
        rows.append(f"| {row['population']} | {row['analysis_loss']} | {CLASS_JA.get(row['classification'], '適用なし')} | {row['radius']} | {row['n']} | " + " | ".join(cells) + " |")
    rows += ["", f"符号付き判定余裕の絶対値が0.05以下のモデル：{len(summary['borderline'])} 件。境界近傍CSVに両フレームの値と元設定IDを保存。",
             f"Hの旧raw中心基準から分類が変わったモデル：{sum(r['class_changed_from_raw_rule'] for r in summary['h_rule_comparison'])} / {len(summary['h_rule_comparison'])} 件（平坦対照を含む）。",
             "Aヒートマップは最頻クラスとその数を示す。同数首位は全該当クラスを明示し斜線表示する。", "",
             "![Aの最頻再構成クラス](a_modal_classes.png)", ""]
    if summary["borderline"]:
        rows += ["境界近傍モデルの全件。余裕は中心基準の左辺 − 右辺（平坦背景は中心輝度 − 0.65）。", "",
                 "| 群 | r | ノイズ・量 | 背景SD | 入れる場所 | 方法 | seed | 過去中心 | 現在中心 | 過去余裕 | 現在余裕 | 分類 | 元ID |",
                 "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|---|"]
        for row in summary["borderline"]:
            arms = "/".join(m["arm"] for m in row["memberships"])
            method = row["method"] if row["method"] != "standard" else row["analysis_loss"]
            is_h = any(m["arm"] == "H" for m in row["memberships"])
            noise = ("なし（背景模様）" if row["background_texture_sd"] else "なし（平坦背景）") if is_h else f"{row['noise_type']} {row['noise_level']:g}"
            placement = "入力・教師で共通" if is_h else row["placement"]
            rows.append(f"| {arms} | {row['radius']} | {noise} | {row['background_texture_sd']:g} | {placement} | {method} | {row['seed']} | {row['center_intensity_previous']:.5f} | {row['center_intensity_current']:.5f} | {row['signed_margin_previous']:.5f} | {row['signed_margin_current']:.5f} | {CLASS_JA[row['classification']]} | {row['source_config_hash'][:12]} |")
        rows.append("")
    changed = [row for row in summary["h_rule_comparison"] if row["class_changed_from_raw_rule"]]
    if changed:
        rows += ["Hで旧raw中心ルールから分類が変わった全件。", "",
                 "| r | 背景SD | 損失 | seed | 旧raw分類 | 新分類 | 過去余裕 | 現在余裕 | 元ID |",
                 "|---:|---:|---|---:|---|---|---:|---:|---|"]
        for row in changed:
            rows.append(f"| {row['radius']} | {row['background_texture_sd']:g} | {row['analysis_loss']} | {row['seed']} | {CLASS_JA[row['raw_comparison_classification']]} | {CLASS_JA[row['classification']]} | {row['signed_margin_previous']:.5f} | {row['signed_margin_current']:.5f} | {row['source_config_hash'][:12]} |")
        rows.append("")
    return "\n".join(rows)


def write_outputs(records, baselines, missing, targets, out=OUT):
    unique_records(records)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = condition_counts(records, targets)
    distributions = probe_distribution(records, baselines)
    borderline = [row for row in records if row["borderline_previous"] or row["borderline_current"]]
    h_rows = [row for row in records if any(m["arm"] == "H" for m in row["memberships"])]
    data = heatmap_data(cells)
    summary = dict(protocol="circle-ae-classification-summary-v1", expected_models=514,
                   measured_models=len(records), complete=not missing, missing=missing,
                   expected_memberships=EXPECTED_MEMBERSHIPS, condition_counts=cells,
                   probe_distributions=distributions, borderline=borderline, h_rule_comparison=h_rows,
                   heatmap_cells=data, untrained_reference=baselines)
    for name, values in (("models", records), ("memberships", expanded_memberships(records)),
                         ("condition_counts", cells), ("probe_distributions", distributions),
                         ("borderline", borderline), ("h_rule_comparison", h_rows),
                         ("untrained_reference", baselines), ("heatmap_cells", data)):
        csv_write(out / f"{name}.csv", values)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (out / "tables.md").write_text(markdown(summary))
    save_heatmap(data, out / "a_modal_classes.png")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    records, baselines, missing, targets = collect(load_ledger(RUNS), load_ledger(LEDGER), device=args.device,
                                                  allow_partial=args.allow_partial)
    result = write_outputs(records, baselines, missing, targets, args.out)
    print(json.dumps(dict(complete=result["complete"], measured_models=len(records),
                         missing=len(missing), output=str(args.out))))


if __name__ == "__main__":
    main()
