"""Re-evaluate the noisy-input models (experiment D: placement input_only / both) with clean probe inputs.

The original evaluation encoded the same (noisy) inputs the model was trained on, so probe R² for these
runs mixes representation quality with input corruption. Here every saved model is loaded without
retraining, its original metrics are reproduced first (sanity), and then the probes are fit and scored
on clean inputs (placement switched to target_only, so the encoder sees the clean frames). Reconstruction
metrics are also reported for the clean input. Nothing in results/runs.jsonl is modified.
"""
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from evaluation import evaluate
from frame_classification import load_saved
from run_sweep import read_results
from training import Config

OUT = Path("results/clean_probe_reeval")
KEYS = ("position_r2_mlp", "velocity_r2_mlp", "future_r2_mlp", "psnr")


def args_for(cfg, row, affine):
    return dict(**cfg.movie_kwargs(), device="cuda", batch_size=cfg.batch_size, n_train=cfg.probe_train,
                n_test=cfg.probe_test, probe_splits=tuple(row.get("probe_splits", (2, 3))), affine=affine)


def main():
    rows = [r for r in read_results().values()
            if r.get("purpose") == "experiment" and r["config"].get("placement") in ("input_only", "both")
            and r["config"]["steps"] == 3000 and r["config"]["lr"] == .003]
    print("models", len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for row in sorted(rows, key=lambda r: json.dumps(r["config"], sort_keys=True)):
        model, cfg, path = load_saved(row, "cuda")
        orig = row["metrics"]
        repro = None
        for affine in (False, True):
            m = evaluate(model, **args_for(cfg, row, affine))
            if all(m[k] == orig[k] for k in KEYS):
                repro = affine
                break
        if repro is None:
            raise ArithmeticError(f"could not reproduce {row['config_hash']}: {[(k, m[k], orig[k]) for k in KEYS]}")
        clean_cfg = replace(cfg, placement="target_only")
        c = evaluate(model, **args_for(clean_cfg, row, repro))
        c_rec = {"placement": cfg.placement, "noise_type": cfg.noise_type, "noise_level": cfg.noise_level,
                 "loss": cfg.loss, "radius": cfg.radius, "seed": cfg.seed,
                 **{f"noisy_input_{k}": orig[k] for k in KEYS}, **{f"clean_input_{k}": c[k] for k in KEYS},
                 "clean_center_previous": c.get("center_intensity_previous"),
                 "clean_center_current": c.get("center_intensity_current"),
                 "config_hash": row["config_hash"], "checkpoint": str(path)}
        records.append(c_rec)
        print(f"{cfg.placement} {cfg.noise_type} {cfg.noise_level:g} {cfg.loss} seed={cfg.seed} "
              f"vel noisy={orig['velocity_r2_mlp']:.3f} clean={c['velocity_r2_mlp']:.3f}", flush=True)
    with (OUT / "seeds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    g = defaultdict(list)
    for x in records:
        g[(x["placement"], x["noise_type"], x["noise_level"], x["loss"])].append(x)
    lines = ["| 場所 | ノイズ | 損失 | n | 位置R² ノイズ入力 → クリーン | 速度R² ノイズ入力 → クリーン | 成功数 ノイズ入力 → クリーン |",
             "|---|---|---|---:|---|---|---|"]
    for k in sorted(g):
        xs = g[k]
        f = lambda key: statistics.fmean(x[key] for x in xs)
        ok = lambda key: sum(x[key] > .8 for x in xs)
        lines.append(f"| {k[0]} | {k[1]} {k[2]:g} | {k[3]} | {len(xs)} | "
                     f"{f('noisy_input_position_r2_mlp'):.3f} → {f('clean_input_position_r2_mlp'):.3f} | "
                     f"{f('noisy_input_velocity_r2_mlp'):.3f} → {f('clean_input_velocity_r2_mlp'):.3f} | "
                     f"{ok('noisy_input_velocity_r2_mlp')}/{len(xs)} → {ok('clean_input_velocity_r2_mlp')}/{len(xs)} |")
    (OUT / "tables.md").write_text("# 入力ノイズ条件のクリーン入力での再評価\n\n保存済みモデルを再学習せずに読み込み、"
                                   "元の指標を再現したうえで、プローブの入力だけをクリーン画像に替えた。\n\n"
                                   + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
