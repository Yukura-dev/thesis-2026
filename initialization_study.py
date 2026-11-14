"""Final K: change initialization only; preserve all historical defaults and hashes."""
import argparse
import json
import math
import os
import time

from filelock import FileLock
import numpy as np
import torch

from collapse_trajectories import append_result
from data import MovieStream
from diagnose_steps import capture_rng, restore_rng, save_checkpoint
from evaluation import ImageMetrics, evaluate
from run_sweep import ROOT, digest, identity, read_results, runtime_info as main_runtime
from training import (Autoencoder, Config, _configure, _parameter_digest, _sync,
                      reconstruction_loss, resolve_device)

PROTOCOL = "circle-ae-final-k-initialization-v1"
PURPOSE = "final_k_initialization"
TRAJECTORY_PURPOSE = "final_k_initialization_trajectory"
FAILURE_PURPOSE = "final_k_initialization_failure"
VARIANTS = ("a", "b")
OBSERVATION_STEPS = (0, 300, 1000, 3000)
SPLITS = (6, 7)
THRESHOLD = .65
GPU_LOCK = ROOT / "checkpoints" / "final_jkl_gpu.lock"
CACHE = ROOT / "checkpoints" / "final_k"
TRAJECTORY = ROOT / "results" / "initialization_trajectory.jsonl"
FAILURES = ROOT / "results" / "initialization_failures.jsonl"


def config_for(radius, noise_level, loss, seed):
    return Config(radius=radius, noise_type="gaussian", noise_level=noise_level,
                  placement="target_only", loss=loss, seed=seed, latent_dim=16,
                  steps=3000, batch_size=128, lr=.003, background=.3, foreground=1.,
                  frame_gap=4, reject_reflections=True)


def plan():
    return [(variant, config_for(radius, noise, loss, seed))
            for variant in VARIANTS for radius in (5, 8) for noise in (0., .4)
            for loss in ("l1", "l2") for seed in range(5)]


def runtime_info(device):
    return {**main_runtime(device, filtered=True), "protocol": PROTOCOL}


def endpoint_spec(cfg, variant, runtime):
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported K initialization: {variant}")
    return identity(cfg, PURPOSE, {**runtime, "initialization": variant})


def trajectory_spec(spec, step):
    return {"config": spec["config"], "purpose": TRAJECTORY_PURPOSE,
            "runtime": {**spec["runtime"], "checkpoint_step": step,
                        "endpoint_config_hash": digest(spec)}}


def failure_spec(spec):
    return {**spec, "purpose": FAILURE_PURPOSE}


def baseline_rows(rows, device):
    """Return the 40 original final models; never reconstruct or retrain a baseline."""
    result = []
    runtime = main_runtime(device, filtered=True)
    for _, cfg in plan()[:40]:
        source = rows.get(digest(identity(cfg, "experiment", runtime)))
        if source is None or source.get("probe_splits") != list(SPLITS):
            raise ValueError(f"Missing exact existing K baseline: {cfg}")
        result.append(source)
    return result


def classification(metrics):
    past = metrics["center_intensity_previous"] >= THRESHOLD
    current = metrics["center_intensity_current"] >= THRESHOLD
    return ("background_only", "current_only", "past_only", "both")[2 * past + current]


def borderline(metrics):
    return {name: abs(metrics[f"center_intensity_{name}"] - THRESHOLD) <= .05
            for name in ("previous", "current")}


def build_model(cfg, variant, device):
    _configure(cfg.seed)
    return Autoencoder(cfg.latent_dim, cfg.background, initialization=variant).to(device)


def folder_for(spec):
    return CACHE / digest(spec)


def load_progress(spec, target):
    candidates = []
    for path in folder_for(spec).glob("*.pt"):
        if path.stem != "latest" and not path.stem.isdecimal():
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["spec"] != spec:
            raise ValueError(f"Wrong K checkpoint identity: {path}")
        if state["step"] <= target:
            candidates.append(state)
    return max(candidates, key=lambda s: s["step"]) if candidates else None


class StudyFailure(RuntimeError):
    def __init__(self, message, observed_step):
        super().__init__(message)
        self.observed_step = observed_step


def train_to(cfg, spec, device, target):
    """Restore full state; retain observation checkpoints and at most 1k lost work."""
    variant = spec["runtime"]["initialization"]
    model = build_model(cfg, variant, device)
    initial_sha = _parameter_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    state = load_progress(spec, target)
    step, elapsed = 0, 0.
    if state is not None:
        if state["initial_model_sha256"] != initial_sha:
            raise ArithmeticError("Initial parameters differ from the saved K trajectory")
        model.load_state_dict(state["model"])
        if _parameter_digest(model) != state["model_sha256"]:
            raise ArithmeticError("K checkpoint parameter checksum mismatch")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state, stream, device)
        stream.accepted, stream.proposed = state["accepted"], state["proposed"]
        step, elapsed = state["step"], state["train_seconds"]
    else:
        state = {"spec": spec, "step": 0, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "model_sha256": initial_sha,
                 "initial_model_sha256": initial_sha, "last_train_loss": None,
                 "train_seconds": 0., "accepted": 0, "proposed": 0,
                 **capture_rng(stream, device)}
        save_checkpoint(folder_for(spec) / "0.pt", state)
    model.train()
    while step < target:
        stop = min((step // 1000 + 1) * 1000, target)
        _sync(device)
        started = time.perf_counter()
        for _ in range(step, stop):
            batch = stream.sample(cfg.batch_size)
            optimizer.zero_grad(set_to_none=True)
            output = model(torch.from_numpy(batch.input).to(device))
            loss = reconstruction_loss(output, torch.from_numpy(batch.target).to(device), cfg.loss)
            loss.backward()
            optimizer.step()
        _sync(device)
        elapsed += time.perf_counter() - started
        step = stop
        last_loss = loss.item()
        if not math.isfinite(last_loss):
            raise StudyFailure("Nonfinite training loss at the fixed checkpoint", step)
        state = {"spec": spec, "step": step, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "model_sha256": _parameter_digest(model),
                 "initial_model_sha256": initial_sha, "last_train_loss": last_loss,
                 "train_seconds": elapsed, "accepted": stream.accepted, "proposed": stream.proposed,
                 **capture_rng(stream, device)}
        path = folder_for(spec) / (f"{step}.pt" if step in OBSERVATION_STEPS else "latest.pt")
        save_checkpoint(path, state)
        print(f"progress K {digest(spec)[:10]} step={step} train={elapsed:.1f}s", flush=True)
    return model, state


def image_observation(model, cfg, device):
    """Fixed clean 5k split 7, with no probe fit at intermediate checkpoints."""
    stats = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
    output_sum = np.zeros(2, dtype=np.float64)
    stream = MovieStream(np.random.SeedSequence([SPLITS[1], 0]), **cfg.movie_kwargs())
    model.eval()
    with torch.inference_mode():
        for start in range(0, cfg.probe_test, cfg.batch_size):
            batch = stream.sample(min(cfg.batch_size, cfg.probe_test-start), need_target=False)
            output = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
            if not np.isfinite(output).all():
                raise FloatingPointError("Nonfinite held-out reconstruction")
            stats.update(output, batch, cfg.radius)
            output_sum += output.sum(axis=(0, 2, 3), dtype=np.float64)
    means = output_sum / (cfg.probe_test * 64 * 64)
    return {**stats.result(), "output_mean": float(means.mean()),
            "output_mean_previous": float(means[0]), "output_mean_current": float(means[1])}


def append_sidecar(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        existing = read_results(path, repair=True).get(row["config_hash"])
        if existing is not None:
            if existing != row:
                raise ValueError(f"Conflicting K sidecar row: {row['config_hash']}")
            return existing
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
            out.flush()
            os.fsync(out.fileno())
    return row


def observe(cfg, spec, model, state):
    point_spec = trajectory_spec(spec, state["step"])
    key = digest(point_spec)
    previous = read_results(TRAJECTORY).get(key)
    if previous is not None:
        if previous["model_sha256"] != state["model_sha256"]:
            raise ArithmeticError("K trajectory weights differ from the cached observation")
        return previous
    metrics = image_observation(model, cfg, next(model.parameters()).device)
    row = {**point_spec, "config_hash": key, "endpoint_config_hash": digest(spec),
           "step": state["step"], "model_sha256": state["model_sha256"],
           "initial_model_sha256": state["initial_model_sha256"], "metrics": metrics,
           "evaluation_seed_sequence": [7, 0], "evaluation_n": cfg.probe_test,
           "frame_classification": classification(metrics), "borderline_frames": borderline(metrics),
           "classification_threshold": THRESHOLD, "train_seconds": state["train_seconds"]}
    return append_sidecar(TRAJECTORY, row)


def run_one(cfg, variant, runtime, device):
    spec = endpoint_spec(cfg, variant, runtime)
    key = digest(spec)
    existing = read_results().get(key)
    if existing is not None:
        if all(digest(trajectory_spec(spec, step)) in read_results(TRAJECTORY)
               for step in OBSERVATION_STEPS):
            return "skip"
        raise ValueError("A completed K endpoint is missing its required trajectory observations")
    if digest(failure_spec(spec)) in read_results(FAILURES):
        return "failed_previous"
    observed_step = 0
    try:
        for observed_step in OBSERVATION_STEPS:
            model, state = train_to(cfg, spec, device, observed_step)
            observation = observe(cfg, spec, model, state)
        started = time.perf_counter()
        scores = evaluate(model, **cfg.movie_kwargs(), device=device, batch_size=cfg.batch_size,
                          n_train=cfg.probe_train, n_test=cfg.probe_test, probe_splits=SPLITS,
                          temporal=True, affine=True)
        temporal = scores.pop("temporal_reconstruction")
        if not all(math.isfinite(v) for v in scores.values()):
            raise FloatingPointError("Nonfinite endpoint metric")
        for name in observation["metrics"].keys() & scores.keys():
            if scores[name] != observation["metrics"][name]:
                raise ArithmeticError(f"Final image measurement differs from trajectory: {name}")
        if state["step"] != 3000 or state["accepted"] != cfg.steps * cfg.batch_size:
            raise ArithmeticError("K training count differs from its fixed budget")
        row = {**spec, "config_hash": key, "first_stage": None, "initialization": variant,
               "baseline_config_hash": digest(identity(cfg, "experiment", main_runtime(device, filtered=True))),
               "device": str(device), "model_sha256": state["model_sha256"],
               "initial_model_sha256": state["initial_model_sha256"], "metrics": scores,
               "temporal_reconstruction": temporal, "last_train_loss": state["last_train_loss"],
               "probe_splits": list(SPLITS), "training_seed_sequence": [0, cfg.seed],
               "train_window_counts": {"accepted": state["accepted"], "proposed": state["proposed"]},
               "data_stream_state_sha256": digest(state["stream_rng"]),
               "frame_classification": classification(scores), "borderline_frames": borderline(scores),
               "classification_threshold": THRESHOLD,
               "trajectory_config_hashes": [digest(trajectory_spec(spec, step)) for step in OBSERVATION_STEPS],
               "timing": {"train_seconds": state["train_seconds"],
                          "evaluation_seconds": time.perf_counter()-started}}
        append_result(row)
        print(f"saved K {key[:10]} {variant} r={cfg.radius} noise={cfg.noise_level} {cfg.loss} "
              f"seed={cfg.seed} velocity={scores['velocity_r2_mlp']:.5f} "
              f"frames={row['frame_classification']}", flush=True)
        return "saved"
    except Exception as exc:
        failed = failure_spec(spec)
        last = load_progress(spec, cfg.steps)
        row = {**failed, "config_hash": digest(failed), "endpoint_config_hash": key,
               "observed_step": getattr(exc, "observed_step", observed_step),
               "last_completed_step": last["step"] if last is not None else None,
               "exception_type": type(exc).__name__, "exception_message": str(exc),
               "status": "failed", "retry_performed": False}
        append_sidecar(FAILURES, row)
        print(f"FAILED K {key[:10]} step={row['observed_step']} {type(exc).__name__}: {exc}", flush=True)
        return "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    runtime = runtime_info(device)
    jobs = plan()
    baseline_rows(read_results(), device)
    rows = read_results()
    done = sum(digest(endpoint_spec(cfg, variant, runtime)) in rows for variant, cfg in jobs)
    print(json.dumps({"requested": len(jobs), "completed": done, "baseline_reused": 40,
                      "runtime": runtime}), flush=True)
    if args.plan:
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    for variant, cfg in jobs:
        with FileLock(GPU_LOCK):
            result = run_one(cfg, variant, runtime, device)
        time.sleep(.15)
        print(f"K condition {result}: {variant} r={cfg.radius} noise={cfg.noise_level} "
              f"{cfg.loss} seed={cfg.seed}", flush=True)
    rows, failures = read_results(), read_results(FAILURES)
    completed = sum(digest(endpoint_spec(cfg, variant, runtime)) in rows for variant, cfg in jobs)
    failed = sum(digest(failure_spec(endpoint_spec(cfg, variant, runtime))) in failures
                 for variant, cfg in jobs)
    print(f"K complete: {completed}/80 endpoints, {failed} failed; no baseline training", flush=True)
    if failed or completed != 80:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
