"""Compare the onset of visible circles at the two contrasts in stage 3 arm C."""
import argparse
import csv
from dataclasses import replace
import json
import math
import os

from filelock import FileLock
import torch

from collapse_replay import CACHE, test_batches
from diagnose_steps import build_model
from evaluation import ImageMetrics
from run_sweep import (LEVELS, ROOT, RUNS, SEEDS, digest, identity, read_results, runtime_info,
                       stage3_arms)
from training import Config, _parameter_digest, resolve_device


PROTOCOL = "circle-ae-stage3-historical-frame-centers-v1"
LEDGER = ROOT / "results" / "stage3_contrast_historical.jsonl"
SEED_CSV = ROOT / "results" / "stage3_contrast_seeds.csv"
ONSET_CSV = ROOT / "results" / "stage3_contrast_onsets.csv"
CENTER_FIELDS = ("center_intensity_previous", "center_intensity_current")
SIGMAS = LEVELS["gaussian"]
FOREGROUNDS = (1., .65)
CHECK_TOL = 1e-4


def conditions():
    base = Config(frame_gap=4, steps=3000, reject_reflections=True)
    low = stage3_arms(base, {"l1": .003, "l2": .003})["C"]
    return [replace(cfg, foreground=fg) for fg in FOREGROUNDS for cfg in low]


def source_row(cfg, rows, rt):
    key = digest(identity(cfg, "experiment", rt))
    if key not in rows:
        raise ValueError(f"Unmeasured source: fg={cfg.foreground}, loss={cfg.loss}, "
                         f"sigma={cfg.noise_level}, seed={cfg.seed}.")
    source = rows[key]
    if source.get("probe_splits") != [6, 7]:
        raise ValueError(f"Wrong test data in source {key}")
    return source


def measurement_spec(source):
    return {"purpose": "stage3_historical_frame_centers", "config": source["config"],
            "runtime": {**source["runtime"], "protocol": PROTOCOL, "source_device": source["device"],
                        "device": "cpu", "cpu_threads": 2, "source_config_hash": source["config_hash"]}}


def archived_centers(source):
    """Evaluate cached stage-1/2 weights, never train a replacement model here."""
    key = source["config_hash"]
    path = CACHE / key / "weights.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived weights for {key}: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**source["config"])
    device = torch.device("cpu")
    model = build_model(cfg, device)
    torch.set_num_threads(2)
    model.load_state_dict(saved["model"])
    if saved["source_config_hash"] != key or _parameter_digest(model) != source["model_sha256"]:
        raise ValueError(f"Model checksum mismatch for {key}")
    model.eval()
    scores = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
    for batch, pred in test_batches(model, cfg, device):
        scores.update(pred, batch, cfg.radius)
    measured = scores.result()
    differences = {name: abs(val - source["metrics"][name]) for name, val in measured.items()
                   if name not in CENTER_FIELDS}
    if any(not math.isfinite(v) or v > CHECK_TOL for v in differences.values()):
        raise ArithmeticError(f"Stage 1/2 reference differs on the fixed 5,000 examples: {key}, {differences}")
    spec = measurement_spec(source)
    return {**spec, "config_hash": digest(spec), "model_sha256": source["model_sha256"],
            "probe_score_stream": [7, 0], "evaluation_pairs": cfg.probe_test,
            "evaluation_device": "cpu", "original_metric_absolute_differences": differences,
            "metrics": measured}


def historical(rt):
    source_rows = read_results(RUNS)
    with FileLock(str(LEDGER) + ".lock"):
        prior = read_results(LEDGER, repair=True)
        for cfg in conditions():
            if cfg.foreground != 1.:
                continue
            source = source_row(cfg, source_rows, rt)
            key = digest(measurement_spec(source))
            if key not in prior:
                row = archived_centers(source)
                with LEDGER.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                prior[key] = row
                print(f"Archived fg=1, sigma={cfg.noise_level:g}, {cfg.loss}, seed={cfg.seed} verified.", flush=True)
    return prior


def visibility(earlier, later, foreground, background):
    threshold = (foreground + background) / 2
    return (threshold, earlier >= threshold, later >= threshold)


def collect(prior, source_rows, rt):
    records = []
    for cfg in conditions():
        source = source_row(cfg, source_rows, rt)
        if cfg.foreground == 1.:
            key = digest(measurement_spec(source))
            replay = prior[key]
            if (replay["model_sha256"] != source["model_sha256"] or replay["probe_score_stream"] != [7, 0] or
                    replay["evaluation_pairs"] != cfg.probe_test):
                raise ValueError(f"Reconstruction reference does not match source: {key}")
            metrics = replay["metrics"]
        else:
            metrics = source["metrics"]
        earlier, later = [metrics[k] for k in CENTER_FIELDS]
        if (not math.isfinite(earlier) or not math.isfinite(later) or
                abs((earlier + later)/2 - source["metrics"]["center_intensity"]) > CHECK_TOL):
            raise ArithmeticError(f"Pooled and channel-specific centers differ: {source['config_hash']}")
        threshold, prev, curr = visibility(earlier, later, cfg.foreground, cfg.background)
        records.append({"foreground": cfg.foreground, "background": cfg.background, "loss": cfg.loss,
                        "sigma": cfg.noise_level, "seed": cfg.seed, "threshold": threshold,
                        CENTER_FIELDS[0]: earlier, CENTER_FIELDS[1]: later,
                        "visible_previous": prev, "visible_current": curr,
                        "visible_either": prev or curr, "visible_both": prev and curr,
                        "psnr": source["metrics"]["psnr"], "velocity_r2_mlp": source["metrics"]["velocity_r2_mlp"],
                        "model_sha256": source["model_sha256"], "source_config_hash": source["config_hash"]})
    return records


def first_and_reversal(pattern):
    """First observed crossing and any later loss of visibility across larger tested sigmas."""
    if len(pattern) != len(SIGMAS):
        raise ValueError("The complete five-sigma design is required to evaluate onset")
    first = next((i for i, present in enumerate(pattern) if present), None)
    return (None, False) if first is None else (SIGMAS[first], not all(pattern[first:]))


def summarize(records):
    index = {(r["foreground"], r["loss"], r["seed"], r["sigma"]): r for r in records}
    if len(records) != len(index) or len(records) != 100:
        raise ValueError("Need exactly one result per contrast/loss/seed/sigma")
    onsets = []
    for fg in FOREGROUNDS:
        for loss in ("l1", "l2"):
            for seed in SEEDS:
                series = [index[fg, loss, seed, sigma] for sigma in SIGMAS]
                row = dict(foreground=fg, loss=loss, seed=seed)
                for name in ("either", "both"):
                    pattern = [r["visible_" + name] for r in series]
                    onset, reversal = first_and_reversal(pattern)
                    row["first_sigma_" + name] = onset
                    row["subsequent_loss_" + name] = reversal
                    row["pattern_" + name] = "".join("1" if present else "0" for present in pattern)
                onsets.append(row)
    return onsets


def export_csv(path, records):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, path)


def markdown(records, onsets):
    lines = ["### コントラストを半分にしたときの円の出現", "",
             "半径5・背景0.3の5,000評価例で、過去フレームと現在フレームの真の中心における再構成輝度を各々平均した。"
             "平均値が背景と物体の中点（物体1.0なら0.65、物体0.65なら0.475）以上で、そのフレームに「円あり」と数えた。"
             "表は同じ5シードで該当した数。σ=0は教師ノイズなし。", "",
             "| 損失 | 物体輝度 | 指標 | σ=0 | 0.05 | 0.1 | 0.2 | 0.4 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for loss in ("l1", "l2"):
        for fg in FOREGROUNDS:
            for metric, label in (("visible_either", "少なくとも片方"), ("visible_both", "両方")):
                numbers = [sum(r[metric] for r in records if r["loss"] == loss and
                               r["foreground"] == fg and r["sigma"] == v) for v in SIGMAS]
                lines.append(f"| {loss.upper()} | {fg:g} | {label} | " + " | ".join(f"{n}/5" for n in numbers) + " |")
    lines += ["", "L1の各シードについて、初めて基準を満たした測定水準を示す。"
              "「なし」は全5水準で未到達。「＊」はそれより高いσのいずれかで基準を割ったことを示す。", "",
              "| 物体輝度 | フレーム | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for fg in FOREGROUNDS:
        selected = sorted((r for r in onsets if r["loss"] == "l1" and r["foreground"] == fg),
                          key=lambda r: r["seed"])
        for name, label in (("either", "少なくとも片方"), ("both", "両方")):
            cells = []
            for r in selected:
                onset = r["first_sigma_" + name]
                cell = "なし" if onset is None else f"{onset:g}"
                cells.append(cell + ("＊" if r["subsequent_loss_" + name] else ""))
            lines.append(f"| {fg:g} | {label} | " + " | ".join(cells) + " |")
    l1_onsets = {(r["seed"], r["foreground"]): r for r in onsets if r["loss"] == "l1"}
    for name, label in (("either", "少なくとも片方"), ("both", "両方")):
        field = "first_sigma_" + name
        pairs = [(l1_onsets[s, 1.][field], l1_onsets[s, .65][field]) for s in SEEDS]
        earlier = same = later = low_only = high_only = neither = 0
        for high, low in pairs:
            if high is None and low is None:
                neither += 1
            elif high is None:
                low_only += 1
            elif low is None:
                high_only += 1
            elif low < high:
                earlier += 1
            elif low > high:
                later += 1
            else:
                same += 1
        lines += ["", f"L1で「{label}」の初回到達をシード内で比較すると、コントラスト半減後に低いσで到達 {earlier}、"
                  f"同じσで到達 {same}、高いσで到達 {later}。半減後だけ到達 {low_only}、半減前だけ到達 {high_only}、"
                  f"両方で未到達 {neither}（全5シード）。"]
    reversed_count = sum(r["subsequent_loss_" + name] for r in onsets for name in ("either", "both"))
    if reversed_count == 0:
        lines += ["", "今回の5水準では、片方・両方の各判定で、一度到達した後に高いσで基準を割るシードはなかった。"]
    else:
        lines += ["", f"一度到達した後に高いσで基準を割った系列は {reversed_count}/40（2損失×2コントラスト×5シード×2判定）だった。"]
    lines.extend(["", "各σでは別のAEを同一シードで初期化して学習したので、この出現点は試したσのうちでの比較。"
                  "[シード・σ別の中心輝度](results/stage3_contrast_seeds.csv) · "
                  "[各シードの初回到達と再低下](results/stage3_contrast_onsets.csv) · "
                  "[元の重みによる高コントラストの再評価](results/stage3_contrast_historical.jsonl)"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--historical-only", action="store_true")
    ap.add_argument("--source-device", default="auto", choices=("auto", "cpu", "mps", "cuda"),
                    help="device recorded in the reference training runs (rescoring uses CPU)")
    args = ap.parse_args()
    rt = runtime_info(resolve_device(args.source_device), filtered=True)
    prior = historical(rt)
    if args.historical_only:
        print("Full-contrast references ready.")
        return
    records = collect(prior, read_results(), rt)
    onsets = summarize(records)
    export_csv(SEED_CSV, records)
    export_csv(ONSET_CSV, onsets)
    print(markdown(records, onsets))


if __name__ == "__main__":
    main()
