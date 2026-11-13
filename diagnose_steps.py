"""Clean r=5 L2 step-budget diagnosis. Train once per seed, evaluate four checkpoints."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from filelock import FileLock
import numpy as np
import torch

from data import MovieStream
from run_sweep import RUNS, digest, historical_config, identity, read_results
from training import (Autoencoder, Config, _configure, _parameter_digest, _sync,
                      reconstruction_loss, resolve_device)

ROOT = Path(__file__).resolve().parent
LEDGER = RUNS
BUDGETS = (300, 1000, 3000, 10000)
SEEDS = (0, 1, 2)
PURPOSE = "step_diagnostic"
RNG_NAMES = ("motion", "input_rng", "target_rng")


def runtime_info(device):
    return {"protocol": "circle-ae-step-diagnostic-v1", "device": str(device),
            "torch": str(torch.__version__), "numpy": np.__version__}


def checkpoint_path(runtime, seed, step):
    path = ROOT / "checkpoints" / "step_diagnostic" / digest(runtime)[:16] / str(seed)
    return path / f"{step}.pt"


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as out:
        torch.save(state, out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, runtime, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["runtime"] != runtime or
            historical_config(Config(**state["config"])) != historical_config(replace(config, steps=state["step"]))):
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    return state


def build_model(config, device):
    _configure(config.seed)
    return Autoencoder(config.latent_dim, config.background).to(device)


def restore_rng(state, stream, device):
    torch.set_rng_state(state["torch_rng"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["device_rng"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["device_rng"], device)
    for name in RNG_NAMES:
        getattr(stream, name).bit_generator.state = state["stream_rng"][name]


def capture_rng(stream, device):
    gpu_rng = None
    if device.type == "mps":
        gpu_rng = torch.mps.get_rng_state()
    elif device.type == "cuda":
        gpu_rng = torch.cuda.get_rng_state(device)
    return {"torch_rng": torch.get_rng_state(), "device_rng": gpu_rng,
            "stream_rng": {name: getattr(stream, name).bit_generator.state for name in RNG_NAMES}}


def original_stage0(config, device):
    for row in read_results(RUNS).values():
        if (row["purpose"] == "experiment" and row["first_stage"] == 0 and
                row["device"] == str(device) and row["config"] == historical_config(config) and
                row["runtime"]["torch"] == str(torch.__version__) and
                row["runtime"]["numpy"] == np.__version__):
            return row
    return None


def train_to(config, device, runtime):
    """Continue from the latest recovery point, keeping immutable evaluation checkpoints."""
    target = checkpoint_path(runtime, config.seed, config.steps)
    if target.exists():
        return
    latest = checkpoint_path(runtime, config.seed, "latest")
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, config.seed]), **config.movie_kwargs())
    step, elapsed = 0, 0.
    # A crash can occur between committing the immutable file and refreshing 'latest'.
    candidates = [p for p in (latest, *(checkpoint_path(runtime, config.seed, n)
                                         for n in BUDGETS if n < config.steps)) if p.exists()]
    if candidates:
        path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        state = load_checkpoint(path, runtime, config)
        step, elapsed = state["step"], state["train_seconds"]
        if step >= config.steps:
            raise ValueError(f"Missing earlier requested checkpoint {target} before {path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state, stream, device)
    model.train()
    print(f"seed {config.seed} training {step} -> {config.steps}", flush=True)
    while step < config.steps:
        stop = min(((step // 500) + 1) * 500, config.steps)
        _sync(device)
        start = time.perf_counter()
        for _ in range(step, stop):
            batch = stream.sample(config.batch_size)
            x = torch.from_numpy(batch.input).to(device)
            y = torch.from_numpy(batch.target).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(x), y, config.loss)
            loss.backward()
            optimizer.step()
        _sync(device)
        elapsed += time.perf_counter() - start
        step = stop
        cfg = replace(config, steps=step)
        model_hash = _parameter_digest(model)
        stage0 = original_stage0(cfg, device) if (config.seed, step) == (0, 300) else None
        if stage0 is not None and model_hash != stage0["model_sha256"]:
            raise ArithmeticError("300-step model differs from historical stage 0")
        if stage0 is not None:
            print(f"historical stage 0 SHA-256 matched: {model_hash}", flush=True)
        last_loss = loss.item()
        if not np.isfinite(last_loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        state = {"runtime": runtime, "config": historical_config(cfg), "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": step, "train_seconds": elapsed,
                 "last_train_loss": last_loss, "model_sha256": model_hash, **capture_rng(stream, device)}
        save_checkpoint(target if step == config.steps else latest, state)
        print(f"seed {config.seed} step {step} cumulative train {elapsed:.1f}s", flush=True)
    # The immutable budget file now supersedes all earlier recovery points.
    latest.unlink(missing_ok=True)


def evaluate_saved(config, device, runtime, rows):
    from diagnostic_probes import evaluate_diagnostic
    spec = identity(config, PURPOSE, runtime)
    key = digest(spec)
    if key in rows:
        return
    path = checkpoint_path(runtime, config.seed, config.steps)
    state = load_checkpoint(path, runtime, config)
    model = build_model(config, device)
    model.load_state_dict(state["model"])
    if _parameter_digest(model) != state["model_sha256"]:
        raise ValueError(f"Checkpoint weights do not match recorded checksum: {path}")
    _sync(device)
    start = time.perf_counter()
    scores = evaluate_diagnostic(model, config, device)
    _sync(device)
    evaluation_seconds = time.perf_counter() - start
    # The old stage-0 run provides an independently trained and evaluated anchor.
    stage0 = original_stage0(config, device) if (config.seed, config.steps) == (0, 300) else None
    if stage0 is not None and (state["model_sha256"] != stage0["model_sha256"] or
                               scores["direct"] != stage0["metrics"]):
        raise ArithmeticError(f"Historical stage 0 mismatch: {scores['direct']} vs {stage0['metrics']}")
    row = {**spec, "config_hash": key, "first_stage": None, "device": str(device),
           "model_sha256": state["model_sha256"],
           "last_train_loss": state["last_train_loss"], "metrics": scores["direct"],
           **{name: value for name, value in scores.items() if name != "direct"},
           "stage0_exact_match": True if stage0 else None,
           "timing": {"train_seconds": state["train_seconds"], "evaluation_seconds": evaluation_seconds}}
    encoded = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with LEDGER.open("ab") as out:
        out.write((encoded + "\n").encode())
        out.flush()
        os.fsync(out.fileno())
    rows[key] = row
    print(f"measured seed {config.seed}, {config.steps} steps: "
          f"velocity MLP R²={scores['direct']['velocity_r2_mlp']:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--train-only", action="store_true")
    phase.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    runtime = runtime_info(device)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LEDGER) + ".lock"):
        rows = read_results(LEDGER, repair=True)
        for steps in BUDGETS:
            for seed in SEEDS:
                cfg = Config(radius=5, loss="l2", lr=0.003, steps=steps, seed=seed)
                if digest(identity(cfg, PURPOSE, runtime)) in rows:
                    continue
                if not args.evaluate_only:
                    train_to(cfg, device, runtime)
                if not args.train_only:
                    evaluate_saved(cfg, device, runtime, rows)


if __name__ == "__main__":
    main()
