"""Final G: matched 6,000-step L2, L2→L1, and L1 trajectories.

python switch_study.py --plan
python switch_study.py --device mps

Only a change from L2 to L1 resets Adam. All branches continue the same data
stream, including independent target-noise draws, from the common 3k L2 state.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import time

from filelock import FileLock
import numpy as np
import torch

import collapse_trajectories as legacy
from data import MovieStream
from diagnose_steps import build_model, capture_rng, restore_rng, save_checkpoint
from evaluation import AffineMoments, ImageMetrics, TemporalReconstruction, evaluate
from run_sweep import ROOT, digest, identity, read_results, runtime_info as main_runtime
from training import Config, _parameter_digest, _sync, reconstruction_loss, resolve_device

PROTOCOL = "circle-ae-final-g-switch-v1"
PURPOSE = "final_g_switch"
ANCHOR_PURPOSE = "final_g_l2_anchor"
METHODS = ("l2_6000", "l2_3000_l1_3000", "l1_6000")
SPLITS = (6, 7)
GPU_LOCK = ROOT / "checkpoints" / "final_gh_gpu.lock"
CACHE = ROOT / "checkpoints" / "final_g"


def runtime_info(device):
    return {**main_runtime(device, filtered=True), "protocol": PROTOCOL}


def plan():
    return [(radius, float(p), seed, method)
            for radius, levels in ((5, (0, .1, .2, .3, .45)), (8, (0, .2, .45)))
            for p in levels for seed in range(5) for method in METHODS]


def conditions():
    return list(dict.fromkeys((r, p, seed) for r, p, seed, _ in plan()))


def config_for(radius, level, seed, *, loss="l2", steps=6000):
    return Config(radius=radius, seed=seed, noise_type="salt_pepper" if level else "gaussian",
                  noise_level=level, placement="target_only", loss=loss, steps=steps,
                  frame_gap=4, reject_reflections=True, lr=.003, latent_dim=16, batch_size=128)


def endpoint_spec(cfg, method, runtime):
    if method not in METHODS:
        raise ValueError(f"Unknown G method: {method}")
    cfg = replace(cfg, steps=6000, loss="l2" if method == "l2_6000" else "l1")
    return identity(cfg, PURPOSE, {**runtime, "method": method})


def anchor_spec(cfg, runtime):
    return identity(replace(cfg, steps=3000, loss="l2"), ANCHOR_PURPOSE, runtime)


def completed(rows, runtime):
    return [rows[digest(spec)] for r, p, seed, method in plan()
            if digest(spec := endpoint_spec(config_for(r, p, seed), method, runtime)) in rows]


def original_row(cfg, rows, device):
    spec = identity(replace(cfg, steps=3000), "experiment", main_runtime(device, filtered=True))
    row = rows.get(digest(spec))
    if row is not None and row.get("probe_splits") != list(SPLITS):
        raise ValueError("Cannot reuse a source with different held-out splits")
    return row


def old_full_state(cfg, rows, device, *, warm=False):
    """Import only complete model+Adam+RNG states with exact original 3k identity."""
    if cfg.noise_level != 0:
        return None, None
    family = legacy.WARM if warm else (legacy.ANCHOR if cfg.loss == "l2" else legacy.LONG)
    if family in (legacy.WARM, legacy.ANCHOR) and cfg.radius != 5:
        return None, None
    if family == legacy.LONG and cfg.radius not in (5, 8):
        return None, None
    original = original_row(replace(cfg, loss="l2" if warm else cfg.loss), rows, device)
    if original is None:
        return None, None
    step = 6000 if warm else 3000
    path = legacy.checkpoint_folder(original, family) / f"{step}.pt"
    if not path.exists():
        return None, None
    state = legacy.load_state(path, family, original)
    if state["step"] != step:
        raise ValueError("Legacy checkpoint step mismatch")
    if step == 3000 and state["model_sha256"] != original["model_sha256"]:
        raise ArithmeticError("Legacy 3k checkpoint does not match the published source")
    source = original
    if warm:
        matches = [row for row in rows.values() if row["purpose"] == legacy.WARM and
                   row.get("source_config_hash") == original["config_hash"] and
                   row.get("model_sha256") == state["model_sha256"]]
        if len(matches) != 1:
            raise ValueError("Need exactly one measured legacy switch endpoint")
        source = matches[0]
        anchor_path = legacy.checkpoint_folder(original, legacy.ANCHOR) / "3000.pt"
        anchor_state = legacy.load_state(anchor_path, legacy.ANCHOR, original)
        state = {**state, "train_seconds": state["train_seconds"] + anchor_state["train_seconds"]}
    provenance = {"kind": "full_training_state", "source_config_hash": source["config_hash"],
                  "model_sha256": state["model_sha256"], "step": step,
                  "checkpoint": str(path.relative_to(ROOT))}
    state = {**state, "provenance": [provenance]}
    return state, source


def folder_for(spec):
    return CACHE / digest(spec)


def saved_state(spec, target):
    folder = folder_for(spec)
    candidates = [p for p in folder.glob("*.pt") if p.stem == "latest" or p.stem.isdecimal()]
    states = []
    for path in candidates:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("spec") != spec:
            raise ValueError(f"Wrong G checkpoint identity: {path}")
        if state["step"] <= target:
            states.append(state)
    return max(states, key=lambda s: s["step"]) if states else None


def restore_training(cfg, device, state=None, *, reset_optimizer=False):
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, foreach=False)
    stream = MovieStream(np.random.SeedSequence([0, cfg.seed]), **cfg.movie_kwargs())
    if state is not None:
        model.load_state_dict(state["model"])
        if _parameter_digest(model) != state["model_sha256"]:
            raise ArithmeticError("Training checkpoint SHA-256 mismatch")
        if not reset_optimizer:
            optimizer.load_state_dict(deepcopy(state["optimizer"]))
        restore_rng(state, stream, device)
        stream.accepted, stream.proposed = state["accepted"], state["proposed"]
    model.train()
    return model, optimizer, stream


def train_to(cfg, spec, device, target, *, initial=None, reset_optimizer=False, reference=None):
    """Resume saved states; a new switch branch alone resets the Adam state."""
    state = saved_state(spec, target)
    continuing = state is not None
    if state is None:
        state = initial
    model, optimizer, stream = restore_training(cfg, device, state,
                                               reset_optimizer=reset_optimizer and not continuing)
    step = state["step"] if state else 0
    elapsed = state["train_seconds"] if state else 0.
    provenance = state.get("provenance", []) if state else []
    start_step = step
    imported = state is not None and state.get("spec") != spec
    if imported:
        state = {**state, "spec": spec, "optimizer": optimizer.state_dict(),
                 "provenance": provenance, **capture_rng(stream, device)}
        save_checkpoint(folder_for(spec) / f"{step}.pt", state)
    while step < target:
        stop = min((step // 1000 + 1) * 1000, target)
        _sync(device)
        started = time.perf_counter()
        for _ in range(step, stop):
            batch = stream.sample(cfg.batch_size)
            x = torch.from_numpy(batch.input).to(device)
            y = torch.from_numpy(batch.target).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(x), y, cfg.loss)
            loss.backward()
            optimizer.step()
        _sync(device)
        elapsed += time.perf_counter() - started
        step = stop
        last_loss = loss.item()
        if not np.isfinite(last_loss):
            raise FloatingPointError(f"Nonfinite G training loss at {step}")
        model_hash = _parameter_digest(model)
        if step == 3000 and reference is not None:
            if model_hash != reference["model_sha256"] or last_loss != reference["last_train_loss"]:
                raise ArithmeticError(f"Reconstructed 3k state differs from {reference['config_hash']}")
        state = {"spec": spec, "step": step, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "model_sha256": model_hash,
                 "last_train_loss": last_loss, "train_seconds": elapsed,
                 "accepted": stream.accepted, "proposed": stream.proposed,
                 "provenance": provenance, **capture_rng(stream, device)}
        save_checkpoint(folder_for(spec) / (f"{step}.pt" if step in (3000, 6000) else "latest.pt"), state)
        print(f"progress {digest(spec)[:10]} {step} train={elapsed:.1f}s", flush=True)
    return model, state, start_step


def image_evaluation(model, cfg, device):
    """Extend already measured probes without fitting them again."""
    model.eval()
    stats = ImageMetrics(include_center_intensity=True, include_frame_centers=True)
    moments, changes = AffineMoments(), TemporalReconstruction()
    with torch.inference_mode():
        stream = MovieStream(np.random.SeedSequence([SPLITS[1], 0]), **cfg.movie_kwargs())
        for start in range(0, cfg.probe_test, cfg.batch_size):
            batch = stream.sample(min(cfg.batch_size, cfg.probe_test-start), need_target=False)
            output = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
            stats.update(output, batch, cfg.radius)
            moments.update(output, batch.clean)
            changes.update(batch.clean, output)
        a, b = moments.coefficients()
        stream = MovieStream(np.random.SeedSequence([SPLITS[1], 0]), **cfg.movie_kwargs())
        sse = 0.
        for start in range(0, cfg.probe_test, cfg.batch_size):
            batch = stream.sample(min(cfg.batch_size, cfg.probe_test-start), need_target=False)
            output = model(torch.from_numpy(batch.input).to(device)).cpu().numpy()
            sse += np.square(a * output.astype(np.float64) + b - batch.clean).sum()
    result = stats.result()
    result.update(corrected_psnr=-10 * math.log10(sse/moments.n), affine_a=a, affine_b=b,
                  affine_n_pixels=moments.n, output_pixel_sd=math.sqrt(moments.sxx/moments.n))
    if result["corrected_psnr"] + 1e-6 < result["psnr"]:
        raise ArithmeticError("Affine correction increased clean test error")
    return result, changes.result()


def measure(cfg, spec, model, state, *, source=None, anchor=None, resumed_at=0):
    if state["step"] != cfg.steps or state["accepted"] != cfg.steps * cfg.batch_size:
        raise ArithmeticError("G endpoint step or accepted-example count differs from the requested budget")
    started = time.perf_counter()
    device = next(model.parameters()).device
    provenance = list(state.get("provenance", []))
    if source is not None:
        if state["model_sha256"] != source["model_sha256"]:
            raise ArithmeticError("Cannot reuse scores from different weights")
        extras, temporal = image_evaluation(model, cfg, device)
        for name in source["metrics"].keys() & extras.keys():
            if source["metrics"][name] != extras[name]:
                raise ArithmeticError(f"Existing metric {name} differs for {source['config_hash']}")
        scores = {**source["metrics"], **extras}
        if source.get("temporal_reconstruction", temporal) != temporal:
            raise ArithmeticError("Existing temporal reconstruction differs")
        provenance.append({"kind": "measured_probes_reused_images_verified",
                           "source_config_hash": source["config_hash"],
                           "model_sha256": source["model_sha256"]})
    else:
        scores = evaluate(model, **cfg.movie_kwargs(), device=device, batch_size=cfg.batch_size,
                          n_train=cfg.probe_train, n_test=cfg.probe_test, probe_splits=SPLITS,
                          temporal=True, affine=True)
        temporal = scores.pop("temporal_reconstruction")
    method = spec["runtime"].get("method")
    row = {**spec, "config_hash": digest(spec), "first_stage": None, "method": method,
           "device": str(device), "model_sha256": state["model_sha256"],
           "metrics": scores, "temporal_reconstruction": temporal,
           "probe_splits": list(SPLITS), "training_seed_sequence": [0, cfg.seed],
           "last_train_loss": state["last_train_loss"], "provenance": provenance,
           "anchor_config_hash": anchor["config_hash"] if anchor else None,
           "anchor_model_sha256": anchor["model_sha256"] if anchor else None,
           "data_stream_state_sha256": digest(state["stream_rng"]),
           "anchor_data_stream_state_sha256": anchor["data_stream_state_sha256"] if anchor else None,
           "optimizer_reset_at": 3000 if method == "l2_3000_l1_3000" else None,
           "continued_motion_and_noise_streams": True,
           "training_phases": ([[0, 3000, "l2"], [3000, 6000, "l1"]]
                               if method == "l2_3000_l1_3000" else [[0, cfg.steps, cfg.loss]]),
           "train_window_counts": {"accepted": state["accepted"], "proposed": state["proposed"]},
           "timing": {"trajectory_train_seconds": state["train_seconds"],
                      "evaluation_seconds": time.perf_counter()-started,
                      "resumed_at_step": resumed_at}}
    row = legacy.append_result(row)
    print(f"saved {row['config_hash'][:10]} {method or 'l2_anchor'} r={cfg.radius} "
          f"p={cfg.noise_level} seed={cfg.seed} velocity={scores['velocity_r2_mlp']:.5f} "
          f"centers={scores['center_intensity_previous']:.4f}/{scores['center_intensity_current']:.4f}", flush=True)
    return row


def run_condition(radius, level, seed, device, runtime):
    """Share the 3k L2 state between both L2-based endpoint methods."""
    rows = read_results()
    cfg = config_for(radius, level, seed)
    specs = {m: endpoint_spec(cfg, m, runtime) for m in METHODS}
    if all(digest(spec) in rows for spec in specs.values()):
        return
    aspec = anchor_spec(cfg, runtime)
    acfg = replace(cfg, steps=3000)
    reference = original_row(acfg, rows, device)
    imported, _ = old_full_state(acfg, rows, device)
    model, astate, resumed = train_to(acfg, aspec, device, 3000, initial=imported, reference=reference)
    anchor = rows.get(digest(aspec))
    if anchor is None:
        anchor = measure(acfg, aspec, model, astate, source=reference, resumed_at=resumed)
        rows[anchor["config_hash"]] = anchor
    elif anchor["model_sha256"] != astate["model_sha256"]:
        raise ArithmeticError("Saved anchor differs from the journal")
    for method in METHODS:
        spec = specs[method]
        if digest(spec) in rows:
            continue
        mcfg = Config(**spec["config"])
        initial, source = None, None
        if method == "l2_3000_l1_3000":
            initial, source = old_full_state(mcfg, rows, device, warm=True)
        if method == "l1_6000":
            initial, _ = old_full_state(mcfg, rows, device)
        elif initial is None:
            initial = {**astate, "provenance": list(astate.get("provenance", [])) +
                       [{"kind": "shared_l2_anchor", "source_config_hash": anchor["config_hash"],
                         "model_sha256": anchor["model_sha256"], "step": 3000}]}
        reference = original_row(replace(mcfg, steps=3000), rows, device) if method == "l1_6000" else None
        # Legacy warm states already contain the correctly reset Adam after the switch.
        reset = method == "l2_3000_l1_3000" and initial["step"] == 3000
        model, state, resumed = train_to(mcfg, spec, device, 6000, initial=initial,
                                         reset_optimizer=reset, reference=reference)
        row = measure(mcfg, spec, model, state, source=source,
                      anchor=anchor if method != "l1_6000" else None, resumed_at=resumed)
        rows[row["config_hash"]] = row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    runtime = runtime_info(device)
    rows = read_results()
    print(json.dumps({"endpoints": len(plan()), "l2_anchors": len(conditions()),
                      "completed": len(completed(rows, runtime)), "runtime": runtime}), flush=True)
    if args.plan:
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    for r, p, seed in conditions():
        with FileLock(GPU_LOCK):
            run_condition(r, p, seed, device, runtime)
        time.sleep(.15)  # Let another queued G/H image or training job acquire the shared lock.
        print(f"condition complete r={r} p={p} seed={seed}; "
              f"endpoints={len(completed(read_results(), runtime))}/120", flush=True)
    print("G complete: all 120 endpoints and 40 shared L2 anchors recorded", flush=True)


if __name__ == "__main__":
    main()
