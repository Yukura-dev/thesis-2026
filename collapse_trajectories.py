"""Clean, four-frame-gap continuations used to diagnose reconstruction collapse."""
import argparse
import csv
from dataclasses import replace
import json
import os
import time

from filelock import FileLock
import numpy as np
import torch

from data import MovieStream
from diagnose_steps import build_model, capture_rng, restore_rng, save_checkpoint
from evaluation import ImageMetrics, evaluate
from run_sweep import ROOT, RUNS, digest, identity, read_results, runtime_info
from training import Config, _parameter_digest, _sync, _validate, reconstruction_loss, resolve_device


PROTOCOL = "circle-ae-collapse-trajectories-v1"
LONG = "collapse_long_l1"
SMALL = "collapse_small_l2"
ANCHOR = "collapse_l2_anchor"
WARM = "collapse_l2_to_l1"
SPLITS = (6, 7)
SEEDS = range(5)
BUDGETS = {LONG: (1000, 3000, 10000, 30000), SMALL: (3000, 10000),
           ANCHOR: (3000,), WARM: (6000,)}
CSV = ROOT / "results" / "collapse_trajectories.csv"
CHECKPOINTS = ROOT / "checkpoints"
SHARED = CHECKPOINTS / "collapse_replay"
GPU_LOCK = CHECKPOINTS / "collapse_gpu.lock"


def plan():
    """Stage-1 checkpoints are shared by the continuations and reconstruction replays."""
    jobs = []
    for seed in SEEDS:
        for radius, family in ((5, LONG), (8, LONG), (2, SMALL), (5, ANCHOR)):
            jobs.extend((radius, seed, family, n) for n in BUDGETS[family] if n <= 3000)
    jobs.extend((5, seed, WARM, 6000) for seed in SEEDS)
    jobs.extend((2, seed, SMALL, 10000) for seed in SEEDS)
    jobs.extend((radius, seed, LONG, n) for n in (10000, 30000)
                for seed in SEEDS for radius in (5, 8))
    return jobs


def config_for(radius, seed, family, steps):
    return Config(radius=radius, seed=seed, frame_gap=4, reject_reflections=True, steps=steps,
                  lr=.003, loss="l1" if family in (LONG, WARM) else "l2")


def original_row(cfg, family, rows, runtime):
    old = replace(cfg, steps=3000, loss="l2" if family == WARM else cfg.loss)
    old_spec = identity(old, "experiment", runtime)
    row = rows.get(digest(old_spec))
    if row is None or row.get("first_stage") != 1 or row.get("probe_splits") != list(SPLITS):
        raise ValueError(f"Missing matching filtered stage-1 result: {old_spec}")
    return row


def append_result(row):
    with FileLock(str(RUNS) + ".lock"):
        existing = read_results(repair=True).get(row["config_hash"])
        if existing is not None:
            if (existing["model_sha256"], existing["metrics"]) != (row["model_sha256"], row["metrics"]):
                raise ValueError(f"Conflicting duplicate measurement: {row['config_hash']}")
            return existing
        with RUNS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return row


def checkpoint_folder(original, family):
    return CHECKPOINTS / "collapse_trajectories" / original["config_hash"] / family


def load_state(path, family, original):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["protocol"], state["family"], state["source_config_hash"]) != (
            PROTOCOL, family, original["config_hash"]):
        raise ValueError(f"Wrong checkpoint protocol or source: {path}")
    return state


def load_progress(folder, family, original, target):
    candidates = [(int(path.stem), path) for path in folder.glob("[0-9]*.pt")
                  if path.stem.isdecimal() and int(path.stem) <= target]
    if candidates:
        step, path = max(candidates)
        previous = load_state(path, family, original)
        if previous["step"] != step:
            raise ValueError(f"Checkpoint step mismatch: {path}")
    else:
        previous = None
    latest = folder / "latest.pt"
    if latest.is_file():
        transient = load_state(latest, family, original)
        if transient["step"] <= target and (previous is None or transient["step"] > previous["step"]):
            previous = transient
    return previous


def fit_checkpoint(cfg, family, original, device):
    """Checkpoint all data and optimizer state. Warm L1 starts with fresh Adam moments."""
    folder = checkpoint_folder(original, family)
    folder.mkdir(parents=True, exist_ok=True)
    state = load_progress(folder, family, original, cfg.steps)
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    step, elapsed = 0, 0.
    warm_parent = family == WARM and state is None
    if warm_parent:
        anchor = checkpoint_folder(original, ANCHOR) / "3000.pt"
        if not anchor.exists():
            fit_checkpoint(replace(cfg, steps=3000, loss="l2"), ANCHOR, original, device)
        state = load_state(anchor, ANCHOR, original)
        if state["model_sha256"] != original["model_sha256"]:
            raise ArithmeticError("Warm start does not match the original L2 stage-1 model")

    if state is not None:
        model.load_state_dict(state["model"])
        if _parameter_digest(model) != state["model_sha256"]:
            raise ArithmeticError("Checkpoint checksum mismatch")
        step = state["step"]
        restore_rng(state, stream, device)
        stream.accepted, stream.proposed = state["accepted"], state["proposed"]
        if not warm_parent:
            optimizer.load_state_dict(state["optimizer"])
            elapsed = state["train_seconds"]
    model.train()
    while step < cfg.steps:
        stop = min(((step // 1000) + 1) * 1000, cfg.steps)
        _sync(device)
        start = time.perf_counter()
        for _ in range(step, stop):
            batch = stream.sample(cfg.batch_size)
            x = torch.from_numpy(batch.input).to(device)
            target = torch.from_numpy(batch.target).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(x), target, cfg.loss)
            loss.backward()
            optimizer.step()
        _sync(device)
        elapsed += time.perf_counter() - start
        step = stop
        last_loss = loss.item()
        if not np.isfinite(last_loss):
            raise FloatingPointError(f"Nonfinite train loss for {family} at step {step}")
        model_hash = _parameter_digest(model)
        if step == 3000 and model_hash != original["model_sha256"]:
            raise ArithmeticError(f"3,000-step checksum differs from stage 1: {original['config_hash']}")
        state = dict(protocol=PROTOCOL, family=family, source_config_hash=original["config_hash"],
                     model_sha256=model_hash, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     accepted=stream.accepted, proposed=stream.proposed, train_seconds=elapsed,
                     step=step, last_train_loss=last_loss, **capture_rng(stream, device))
        path = folder / (f"{step}.pt" if step in BUDGETS[family] else "latest.pt")
        save_checkpoint(path, state)
    return model, state


def background_validation(cfg):
    """The fixed 1,024 clean validation movies, rebatching exactly as _validate."""
    stream = MovieStream(np.random.SeedSequence([1, 0]), **cfg.movie_kwargs())
    absolute_error = 0.
    for start in range(0, 1024, cfg.batch_size):
        batch = stream.sample(min(cfg.batch_size, 1024-start), need_target=False)
        absolute_error += np.abs(np.subtract(batch.clean, np.float32(cfg.background), dtype=np.float64)).sum()
    return float(absolute_error / (1024 * 2 * 64**2))


def measure(cfg, family, original, runtime, device, key, spec):
    print(f"{family} r={cfg.radius} seed={cfg.seed} total_step={cfg.steps}", flush=True)
    model, state = fit_checkpoint(cfg, family, original, device)
    start = time.perf_counter()
    validation_l1 = _validate(model, replace(cfg, loss="l1"), device)
    scores = evaluate(model, **cfg.movie_kwargs(), device=device, batch_size=cfg.batch_size,
                      n_train=cfg.probe_train, n_test=cfg.probe_test, probe_splits=SPLITS, temporal=True)
    temporal = scores.pop("temporal_reconstruction")
    _sync(device)
    matches = cfg.steps == 3000 and family != WARM
    if matches:
        if not (state["model_sha256"] == original["model_sha256"] and
                state["last_train_loss"] == original["last_train_loss"] and scores == original["metrics"] and
                state["proposed"] == original["train_window_counts"]["proposed"] and
                state["accepted"] == original["train_window_counts"]["accepted"]):
            raise ArithmeticError(f"3k weights, train stream or test scores differ: {original['config_hash']}")
        if cfg.loss == "l1" and validation_l1 != original["validation_loss"]:
            raise ArithmeticError("L1 validation values differ from stage 1")
        cache = SHARED / original["config_hash"] / "weights.pt"
        payload = {"source_config_hash": original["config_hash"], "model_sha256": state["model_sha256"],
                   "model": {k: v.detach().cpu() for k,v in model.state_dict().items()}}
        save_checkpoint(cache, payload)
    row = {**spec, "config_hash": key, "source_config_hash": original["config_hash"],
           "source_model_sha256": original["model_sha256"], "first_stage": None,
           "device": str(device), "model_sha256": state["model_sha256"],
           "probe_splits": list(SPLITS), "training_seed_sequence": [0, cfg.seed],
           "validation_n": 1024, "validation_seed_sequence": [1, 0],
           "validation_l1": validation_l1, "validation_background_l1": background_validation(cfg),
           "metrics": scores, "temporal_reconstruction": temporal,
           "last_train_loss": state["last_train_loss"], "stage1_exact_match": matches,
           "optimizer_reset_at": 3000 if family == WARM else None,
           "l2_initialization_steps": 3000 if family == WARM else 0,
           "train_window_counts": {"accepted": state["accepted"], "proposed": state["proposed"]},
           "timing": {"train_seconds": state["train_seconds"], "evaluation_seconds": time.perf_counter()-start}}
    row = append_result(row)
    print(f"saved {key[:10]}: center={scores['center_intensity']:.6f} "
          f"PSNR={scores['psnr']:.4f} velocity R²={scores['velocity_r2_mlp']:.5f}", flush=True)
    return row


def flat_record(row):
    metrics = row["metrics"]
    return {"family": row["purpose"], "radius": row["config"]["radius"], "seed": row["config"]["seed"],
            "total_steps": row["config"]["steps"], "loss": row["config"]["loss"],
            "l2_initialization_steps": row["l2_initialization_steps"],
            "psnr": metrics["psnr"], "center_intensity": metrics["center_intensity"],
            "center_contrast": metrics["center_contrast"], "object_mae": metrics["object_mae"],
            "background_mae": metrics["background_mae"], "position_r2_mlp": metrics["position_r2_mlp"],
            "velocity_r2_mlp": metrics["velocity_r2_mlp"],
            "temporal_rms_ratio": row["temporal_reconstruction"]["rms_ratio"],
            "validation_l1": row["validation_l1"], "validation_background_l1": row["validation_background_l1"],
            "source_config_hash": row["source_config_hash"], "stage1_exact_match": row["stage1_exact_match"],
            "model_sha256": row["model_sha256"], "config_hash": row["config_hash"]}


def frame_centers(row, original, device):
    """Re-score the exact held-out movies by channel, to detect a single missing disk."""
    cfg = Config(**row["config"])
    folder = checkpoint_folder(original, row["purpose"])
    cache = folder / f"centers-{cfg.steps}.json"
    if cache.exists():
        measured = json.loads(cache.read_text())
        if measured["model_sha256"] != row["model_sha256"] or measured["config_hash"] != row["config_hash"]:
            raise ValueError(f"Wrong per-frame measurements: {cache}")
        return measured["centers"]
    with FileLock(GPU_LOCK):
        state = load_state(folder / f"{cfg.steps}.pt", row["purpose"], original)
        model = build_model(cfg, device)
        model.load_state_dict(state["model"])
        if _parameter_digest(model) != row["model_sha256"]:
            raise ArithmeticError(f"Checkpoint weights differ before per-frame evaluation: {row['config_hash']}")
        model.eval()
        stream = MovieStream(np.random.SeedSequence([SPLITS[1], 0]), **cfg.movie_kwargs())
        image_stats = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
        with torch.inference_mode():
            for start in range(0, cfg.probe_test, cfg.batch_size):
                batch = stream.sample(min(cfg.batch_size, cfg.probe_test-start), need_target=False)
                pred = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
                image_stats.update(pred, batch, cfg.radius)
        scores = image_stats.result()
        centers = {name: scores.pop(name) for name in ("center_intensity_previous", "center_intensity_current")}
        if any(row["metrics"][k] != value for k, value in scores.items()):
            raise ArithmeticError(f"Held-out image metrics differ before separating frames: {row['config_hash']}")
        measured = {"centers": centers, "model_sha256": row["model_sha256"], "config_hash":row["config_hash"]}
        temp = cache.with_suffix(".tmp")
        temp.write_text(json.dumps(measured, allow_nan=False))
        os.replace(temp, cache)
    return centers


def export(rows, runtime):
    records = []
    for radius, seed, family, steps in plan():
        key = digest(identity(config_for(radius, seed, family, steps), family, runtime))
        if key not in rows:
            raise ValueError(f"Measurements incomplete: missing {key}")
        row = rows[key]
        records.append({**flat_record(row), **frame_centers(row, rows[row["source_config_hash"]],
                                                           resolve_device(runtime["device"]))})
    temp = CSV.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, CSV)
    print(f"Saved {len(records)} measurements in {CSV}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    original_runtime = runtime_info(device, filtered=True)
    runtime = {**original_runtime, "protocol": PROTOCOL}
    rows = read_results()
    if not args.export_only:
        CHECKPOINTS.mkdir(exist_ok=True)
        for radius, seed, family, steps in plan():
            cfg = config_for(radius, seed, family, steps)
            spec = identity(cfg, family, runtime)
            key = digest(spec)
            if key in rows:
                continue
            original = original_row(cfg, family, rows, original_runtime)
            with FileLock(GPU_LOCK):
                latest = read_results()
                if key in latest:
                    rows[key] = latest[key]
                else:
                    rows[key] = measure(cfg, family, original, runtime, device, key, spec)
            time.sleep(.15)  # Let a waiting replay/evaluation use the shared accelerator.
    export(rows, runtime)


if __name__ == "__main__":
    main()
