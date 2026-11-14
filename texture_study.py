"""H: deterministic smooth background textures, with paired clean two-frame targets."""
import argparse
import csv
import json
import os
import time
from dataclasses import replace

from filelock import FileLock

from data import TEXTURE_METHOD
from diagnose_steps import save_checkpoint
from run_sweep import ROOT, RUNS, config_record, digest, read_results, runtime_info as main_runtime
from training import Config, resolve_device, run_experiment

PURPOSE = "background_texture"
PROTOCOL = "circle-ae-background-texture-v1"
SPLITS = (6, 7)
GPU_LOCK = ROOT / "checkpoints" / "final_gh_gpu.lock"
CSV = RUNS.parent / "texture_study_seeds.csv"


def configs():
    base = Config(frame_gap=4, steps=3000, lr=.003, reject_reflections=True)
    return [replace(base, radius=radius, loss=loss, background_texture_sd=sd, seed=seed)
            for radius in (5, 8) for sd in (.1, .4) for loss in ("l1", "l2") for seed in range(5)]


def runtime_info(device):
    return {**main_runtime(device, filtered=True), "protocol": PROTOCOL,
            "texture_method": TEXTURE_METHOD}


def experiment_spec(config, runtime):
    return {"config": config_record(config), "purpose": PURPOSE, "runtime": runtime}


def export(rows, runtime, path=CSV):
    records = []
    for cfg in configs():
        key = digest(experiment_spec(cfg, runtime))
        if key not in rows:
            continue
        row = rows[key]
        records.append({"radius": cfg.radius, "background_texture_sd": cfg.background_texture_sd,
                        "loss": cfg.loss, "seed": cfg.seed, **row["metrics"],
                        "train_seconds": row["timing"]["train_seconds"],
                        "evaluation_seconds": row["timing"]["evaluation_seconds"],
                        "total_seconds": row["timing"]["total_seconds"],
                        "validation_loss": row["validation_loss"],
                        "model_sha256": row["model_sha256"], "config_hash": key})
    if records:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        os.replace(temp, path)
    return len(records)


def run_one(config, runtime, path=RUNS, gpu_lock=GPU_LOCK):
    spec = experiment_spec(config, runtime)
    key = digest(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        saved = read_results(path, repair=True).get(key)
    if saved is not None:
        print(f"skip {key[:10]}", flush=True)
        return saved
    # G and H hold this common lock for the full fit and evaluation, avoiding
    # simultaneous fits on the same accelerator. The ledger lock is held briefly.
    with FileLock(str(gpu_lock)):
        with FileLock(str(path) + ".lock"):
            saved = read_results(path, repair=True).get(key)
        if saved is not None:
            return saved
        print(f"run H {key[:10]} r={config.radius} sd={config.background_texture_sd} "
              f"{config.loss} seed={config.seed}", flush=True)
        cache = ROOT / "checkpoints" / "background_texture" / key / "weights.pt"

        def persist(model):
            save_checkpoint(cache, {"source_config_hash": key,
                                    "model": {name: tensor.detach().cpu()
                                              for name, tensor in model.state_dict().items()}})

        result = run_experiment(config, runtime["device"], probe_splits=SPLITS, affine=True,
                                store_weights=persist)
        row = {**spec, "config_hash": key, "first_stage": None,
               "training_seed_sequence": [0, config.seed],
               "evaluation_reference": "full textured clean image; PSNR fixed data range 1",
               **result}
        with FileLock(str(path) + ".lock"):
            rows = read_results(path, repair=True)
            if key not in rows:
                with path.open("a") as out:
                    out.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
            else:
                row = rows[key]
        print(f"saved H {key[:10]} PSNR={row['metrics']['psnr']:.4f} "
              f"velocity={row['metrics']['velocity_r2_mlp']:.5f} "
              f"in {row['timing']['total_seconds']:.1f}s", flush=True)
    time.sleep(.15)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    runtime = runtime_info(resolve_device(args.device))
    cases = configs()
    rows = read_results()
    count = sum(digest(experiment_spec(cfg, runtime)) in rows for cfg in cases)
    print(f"H: {len(cases)} configurations, {count} complete, {len(cases)-count} remaining; "
          "texture normalization before overlay; no clipping", flush=True)
    if args.dry_run:
        return
    if not args.export_only:
        for cfg in cases:
            run_one(cfg, runtime)
    count = export(read_results(), runtime)
    print(f"Exported {count}/{len(cases)} H rows to {CSV}", flush=True)


if __name__ == "__main__":
    main()
