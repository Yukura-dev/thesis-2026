"""Measure stage-1 readout from the seed-matched encoder at initialization, before any AE update."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from statistics import mean, median, stdev
import tempfile

from filelock import FileLock

from diagnose_steps import build_model
from evaluation import evaluate
from run_sweep import RADII, ROOT, RUNS, SEEDS, digest, read_results, runtime_info
from training import Config, _parameter_digest, resolve_device


PURPOSE = "untrained_encoder"
PROTOCOL = "circle-ae-untrained-encoder-v1"
SPLITS = (6, 7)
PROBE_METRICS = tuple(f"{target}_r2_{probe}" for probe in ("ridge", "mlp")
                      for target in ("position", "velocity", "future"))
GPU_LOCK = ROOT / "checkpoints" / "collapse_gpu.lock"


def experiment_spec(config, runtime):
    # Steps/loss/rate describe training elsewhere. No optimizer or reconstruction
    # objective is constructed in this experiment. Keep the actual update count in its hash.
    record = asdict(config)
    if record.get("background_texture_sd") == 0:
        del record["background_texture_sd"]
    del record["loss"], record["lr"]
    record["steps"] = 0
    return {"config": record, "runtime": runtime, "purpose": PURPOSE}


def measure(config, device):
    model = build_model(config, device)
    model.requires_grad_(False)
    before = _parameter_digest(model)
    metrics = evaluate(model, **config.movie_kwargs(), device=device, batch_size=config.batch_size,
                       n_train=config.probe_train, n_test=config.probe_test, probe_splits=SPLITS)
    if before != _parameter_digest(model):
        raise ValueError("Untrained autoencoder weights changed during probe training")
    return {"model_sha256": before, "probe_splits": list(SPLITS),
            "metrics": {k: metrics[k] for k in PROBE_METRICS}}


def write_csv(path, records):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp.name, path)


def export(measured):
    seeds = sorted(({"radius": r["config"]["radius"], "seed": r["config"]["seed"],
                     **{k: r["metrics"][k] for k in PROBE_METRICS},
                     "model_sha256": r["model_sha256"], "config_hash": r["config_hash"]}
                    for r in measured), key=lambda r: (r["radius"], r["seed"]))
    summaries = []
    for radius in RADII:
        at_radius = [r for r in seeds if r["radius"] == radius]
        if [r["seed"] for r in at_radius] != list(SEEDS):
            raise ValueError(f"Incomplete seeds at radius {radius}")
        for metric in PROBE_METRICS:
            scores = [r[metric] for r in at_radius]
            summaries.append({"radius": radius, "metric": metric, "n_seeds": len(scores),
                              "mean": mean(scores), "sd": stdev(scores), "median": median(scores),
                              "fraction_gt_0_8": sum(s > .8 for s in scores) / len(scores)})
    write_csv(RUNS.parent / "untrained_encoder_seeds.csv", seeds)
    write_csv(RUNS.parent / "untrained_encoder_summary.csv", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    args = parser.parse_args()
    device = resolve_device(args.device)
    runtime = {**runtime_info(device, filtered=True), "protocol": PROTOCOL}
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    measured = []
    for radius in RADII:
        for seed in SEEDS:
            cfg = Config(frame_gap=4, radius=radius, seed=seed, reject_reflections=True)
            spec = experiment_spec(cfg, runtime)
            key = digest(spec)
            with FileLock(str(RUNS) + ".lock"):
                saved = read_results(RUNS, repair=True).get(key)
            if saved is None:
                with FileLock(str(GPU_LOCK)):
                    result = measure(cfg, device)
                row = {**spec, **result, "device": str(device), "config_hash": key, "first_stage": None}
                with FileLock(str(RUNS) + ".lock"):
                    saved = read_results(RUNS, repair=True).get(key)
                    if saved is None:
                        with RUNS.open("a") as stream:
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        saved = row
                print(f"r={radius} seed={seed} velocity MLP R²={saved['metrics']['velocity_r2_mlp']:.5f}",
                      flush=True)
            measured.append(saved)
    if len({r["model_sha256"] for r in measured}) != len(SEEDS):
        raise ValueError("The same initial model for each seed must be used across all radii")
    export(measured)
    print(f"{len(measured)} measured untrained encoders; seed and radius tables in {RUNS.parent}")


if __name__ == "__main__":
    main()
