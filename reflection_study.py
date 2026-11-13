"""Re-score the original 60 unfiltered gap models on the same test trajectories by wall hit."""
import argparse
import json
import os
from pathlib import Path
import time

from filelock import FileLock
import numpy as np
import torch

from data import MovieStream
from diagnose_steps import (build_model, checkpoint_path, load_checkpoint,
                            runtime_info as step_runtime, save_checkpoint)
from evaluation import ImageMetrics, grouped_r2, predict_probes
from gap_study import SELECTION, runtime_info as gap_runtime, selection_configs, study_spec
from run_sweep import RUNS, digest, read_results
from training import _parameter_digest, _sync, reconstruction_loss, resolve_device

ROOT = Path(__file__).resolve().parent
PURPOSE = "reflection_stratification"
PROTOCOL = "circle-ae-unfiltered-reflection-v1"


def weights_for(cfg, source, device, cache_path):
    """Load the three surviving diagnostic checkpoints; replay other missing weights."""
    model = build_model(cfg, device)
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved["source_config_hash"] != source["config_hash"]:
            raise ValueError(f"Cache/source mismatch: {cache_path}")
        model.load_state_dict(saved["model"])
        provenance = "replay_cache"
    else:
        legacy_path = checkpoint_path(step_runtime(device), cfg.seed, cfg.steps)
        from_step = cfg.frame_gap == 1 and cfg.radius == 5 and cfg.seed in (0, 1, 2)
        if from_step and legacy_path.exists():
            state = load_checkpoint(legacy_path, step_runtime(device), cfg)
            model.load_state_dict(state["model"])
            provenance = "historical_checkpoint"
        else:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
            stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
            model.train()
            for _ in range(cfg.steps):
                batch = stream.sample(cfg.batch_size)
                x = torch.from_numpy(batch.input).to(device)
                y = torch.from_numpy(batch.target).to(device)
                opt.zero_grad(set_to_none=True)
                loss = reconstruction_loss(model(x), y, cfg.loss)
                loss.backward()
                opt.step()
            _sync(device)
            if loss.item() != source["last_train_loss"]:
                raise ArithmeticError(f"Final replay loss differs: {source['config_hash']}")
            provenance = "deterministic_replay"
    if _parameter_digest(model) != source["model_sha256"]:
        raise ArithmeticError(f"Replay weights differ from original: {source['config_hash']}")
    if not cache_path.exists():
        save_checkpoint(cache_path, {"model": model.state_dict(),
                                     "source_config_hash": source["config_hash"]})
    return model, provenance


def evaluate_unfiltered(model, cfg, device, source, path):
    from reflection_metrics import historical_geometry, stratified_velocity
    test_images = ImageMetrics()
    model.eval()

    def collect(split, count):
        stream = MovieStream(np.random.SeedSequence([split, 0]), **cfg.movie_kwargs())
        features, labels = [], []
        with torch.inference_mode():
            for start in range(0, count, cfg.batch_size):
                batch = stream.sample(min(cfg.batch_size, count - start), need_target=False)
                z = model.encode(torch.from_numpy(batch.input).to(device))
                features.append(z.cpu().numpy())
                labels.append(batch.labels)
                if split == 3:
                    test_images.update(model.decode(z).cpu().numpy(), batch, cfg.radius)
        return np.concatenate(features), np.concatenate(labels)

    z, y = collect(2, cfg.probe_train)
    z_test, y_test = collect(3, cfg.probe_test)
    pred = predict_probes(z, y, z_test)
    scores = test_images.result()
    for k, p in pred.items():
        scores.update(grouped_r2(y_test, p, k))
    if scores != source["metrics"]:
        raise ArithmeticError(f"Original ten test metrics differ: {source['config_hash']}, {scores}")
    geometry = historical_geometry(cfg.radius, cfg.frame_gap, split=3, count=cfg.probe_test)
    truth = y_test[:, 2:4]
    if (not np.array_equal(truth, geometry["velocity"].astype(np.float32)) or
            not np.array_equal(y_test[:, :2], geometry["centers"][:, 1].astype(np.float32))):
        raise ArithmeticError("Test labels and exact geometry stream do not align")
    hit_axes = geometry["hit_axes"]
    groups = {method: stratified_velocity(truth, p[:, 2:4], hit_axes) for method, p in pred.items()}
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as out:
        np.savez_compressed(out, truth=truth, hit_axes=hit_axes,
                            ridge=pred["ridge"][:, 2:4], mlp=pred["mlp"][:, 2:4],
                            source_config_hash=source["config_hash"])
    os.replace(tmp, path)
    return scores, groups, int(hit_axes.any(axis=1).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    old_runtime = gap_runtime(device)
    runtime = {**old_runtime, "protocol": PROTOCOL}
    with FileLock(str(RUNS.parent / "reflection_study.lock")):
        with FileLock(str(RUNS) + ".lock"):
            rows = read_results(RUNS, repair=True)
        for cfg in selection_configs():
            spec = study_spec(cfg, PURPOSE, runtime)
            key = digest(spec)
            if key in rows:
                continue
            source_key = digest(study_spec(cfg, SELECTION, old_runtime))
            if source_key not in rows:
                raise ValueError(f"Run the historical unfiltered gap selection first: {source_key}")
            source = rows[source_key]
            cache = ROOT / "checkpoints" / "reflection" / source_key
            print(f"stratify Δ={cfg.frame_gap} r={cfg.radius} seed={cfg.seed}", flush=True)
            start = time.monotonic()
            model, provenance = weights_for(cfg, source, device, cache / "weights.pt")
            scores, groups, hits = evaluate_unfiltered(model, cfg, device, source, cache / "predictions.npz")
            row = {**spec, "config_hash": key, "source_config_hash": source_key, "device": str(device),
                   "first_stage": None, "metrics": scores, "stratified": groups,
                   "n_test": cfg.probe_test, "n_reflecting": hits, "reflection_fraction": hits / cfg.probe_test,
                   "weight_provenance": provenance, "model_sha256": source["model_sha256"],
                   "ten_metrics_exact_match": True, "seconds": time.monotonic() - start}
            with FileLock(str(RUNS) + ".lock"):
                with RUNS.open("ab") as out:
                    out.write((json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode())
                    out.flush()
                    os.fsync(out.fileno())
            rows[key] = row
            print(f"saved {key[:10]} exact match, {hits} / {cfg.probe_test} reflecting, "
                  f"MLP R² {groups['mlp']['reflecting']['r2']:.4f} reflecting, "
                  f"{groups['mlp']['nonreflecting']['r2']:.4f} nonreflecting", flush=True)


if __name__ == "__main__":
    main()
