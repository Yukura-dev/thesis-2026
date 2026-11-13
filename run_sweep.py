"""Run one experiment stage, recording only completed training runs."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile

from filelock import FileLock

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs.jsonl"
REPORT = ROOT / "REPORT.md"
RADII = (1, 2, 3, 5, 8)
LEVELS = {"salt_pepper": (0., .05, .1, .2, .3, .45), "gaussian": (0., .05, .1, .2, .4)}
LOSSES = ("l1", "l2", "huber", "charbonnier")
SEEDS = (0, 1, 2, 3, 4)
STAGE3_SEEDS = SEEDS[:3]
PILOT_SEED = 1729
RATES = (3e-4, 1e-3, 3e-3)
PROTOCOL = "circle-ae-v1"  # historical stage 0, calibration and 2/3 probe splits
MAIN_PROTOCOL = "circle-ae-final-gap-v1"  # independent 6/7 probe splits
STAGE0_STEPS = 300
SWEEP_STEPS = 3000


def canonical(config):
    if config.noise_level == 0:
        return replace(config, noise_type="gaussian", placement="target_only", noise_level=0.)
    return config


def stage_configs(stage, base, rates, *, shared_lr=False):
    """Return unique configs in execution order; zero noise has one identity."""
    if shared_lr and stage != 1:
        raise ValueError("Shared LR sensitivity is defined only for stage 1")
    if stage == 0:
        return [replace(base, radius=5, lr=rates["l2"], loss="l2", seed=0)]
    if stage == 1:
        losses = LOSSES if shared_lr else ("l1", "l2")
        if shared_lr:
            rates = dict.fromkeys(LOSSES, 1e-3)
        cells = [(r, "gaussian", 0., "target_only", losses) for r in RADII]
    elif stage in (2, 3):
        radii = (5,) if stage == 2 else RADII
        cells = [(r, kind, level, "target_only", ("l1", "l2"))
                 for kind, levels in LEVELS.items() for level, r in product(levels, radii)]
        if stage == 3:
            for kind in LEVELS:
                cells.append((5, kind, .2, "target_only", ("huber",)))
                for placement in ("input_only", "both"):
                    cells.append((5, kind, .2, placement, ("l1", "l2")))
    else:
        raise ValueError("stage must be 0, 1, 2 or 3")
    seeds = STAGE3_SEEDS if stage == 3 else SEEDS
    configs = (canonical(replace(base, radius=r, noise_type=kind, noise_level=level,
                                 placement=placement, loss=loss, lr=rates[loss], seed=seed))
               for r, kind, level, placement, losses in cells for loss, seed in product(losses, seeds))
    return list(dict.fromkeys(configs))


def calibration_configs(base):
    return [replace(base, radius=5, noise_type="gaussian", noise_level=0., placement="target_only",
                    loss=loss, seed=PILOT_SEED, lr=lr) for loss, lr in product(LOSSES, RATES)]


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def historical_config(config):
    """An omitted gap denotes one frame in version-1 rows and checkpoints."""
    record = asdict(config)
    if record.get("frame_gap") == 1:
        record.pop("frame_gap")
    return record


def identity(config, purpose, runtime):
    config = canonical(config)
    record = asdict(config) if runtime.get("protocol") == MAIN_PROTOCOL else historical_config(config)
    return {"config": record, "purpose": purpose, "runtime": runtime}


def read_results(path=RUNS, repair=False):
    """A torn final append can be retried. Reject corruption of any complete line."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("r+b" if repair else "rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError(f"{path}: invalid complete record (byte {start})")
                if repair:
                    stream.truncate(start)
                break
            if not line.endswith(b"\n") and repair:
                stream.seek(0, 2)
                stream.write(b"\n")
            key = row["config_hash"]
            if key != digest({k: row[k] for k in ("config", "purpose", "runtime")}):
                raise ValueError(f"Invalid configuration hash at byte {start}")
            if key in rows:
                raise ValueError(f"Duplicate run {key}")
            rows[key] = row
    return rows


def run_one(config, purpose, runtime, stage, rows, path):
    from training import run_experiment

    spec = identity(config, purpose, runtime)
    key = digest(spec)
    if key in rows:
        print(f"skip {key[:10]} {purpose}", flush=True)
        return
    print(f"run {key[:10]} {purpose} r={config.radius} {config.loss} lr={config.lr} "
          f"{config.noise_type}={config.noise_level} {config.placement} seed={config.seed}", flush=True)
    kwargs = {"probe_splits": (6, 7)} if runtime["protocol"] == MAIN_PROTOCOL else {}
    outcome = run_experiment(config, device=runtime["device"], calibration=(purpose == "calibration"), **kwargs)
    row = {**spec, "config_hash": key, "first_stage": stage, **outcome}
    line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("ab") as stream:
        stream.write((line + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    rows[key] = row
    print(f"saved {key[:10]} in {outcome['timing']['total_seconds']:.1f} s", flush=True)


def calibration_table(base, runtime, rows):
    by_loss = {loss: {} for loss in LOSSES}
    for config in calibration_configs(base):
        row = rows.get(digest(identity(config, "calibration", runtime)))
        if row is None:
            return None
        score = row["validation_loss"]
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite validation loss in {row['config_hash']}")
        by_loss[config.loss][config.lr] = score
    return by_loss


def select_rates(table):
    return {loss: min(RATES, key=lambda lr: (scores[lr], lr)) for loss, scores in table.items()}


def update_calibration_report(path, base, runtime, table):
    # Keep the human-written stage 0 report and other completed protocols intact.
    name = digest(identity(base, "calibration-summary", runtime))[:12]
    start = f"<!-- calibration {name} -->"
    end = f"<!-- end calibration {name} -->"
    chosen = select_rates(table)
    lines = [start, "### 学習率の選定", "",
             f"r=5、ノイズなし、seed={PILOT_SEED}、{base.steps} ステップ、"
             f"潜在 {base.latent_dim} 次元、{runtime['device']}。独立した固定検証データで、"
             "各行の損失が最小の学習率を採用。同値のときは小さい学習率を選ぶ。", "",
             "| 損失 | 3e-4 | 1e-3 | 3e-3 | 採用した学習率 |",
             "|---|---:|---:|---:|---:|"]
    for loss in LOSSES:
        scores = " | ".join(f"{table[loss][lr]:.9g}" for lr in RATES)
        lines.append(f"| {loss} | {scores} | {chosen[loss]:g} |")
    block = "\n".join([*lines, end])
    previous = path.read_text() if path.exists() else "# 実験報告\n"
    if start in previous:
        a, rest = previous.split(start, 1)
        _, b = rest.split(end, 1)
        updated = a + block + b
    else:
        updated = previous.rstrip() + "\n\n" + block + "\n"
    if updated != previous:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as temp:
            temp.write(updated)
        os.replace(temp.name, path)


def add_design_arguments(parser):
    parser.add_argument("--frame-gap", type=int, choices=(1, 2, 4, 8),
                        help="explicit interval override; stages otherwise require a qualified gap")
    parser.add_argument("--latent-dim", type=int, choices=(4, 16, 64), default=16)
    parser.add_argument("--steps", type=int, default=None,
                        help=f"default: {SWEEP_STEPS} (stage 0: {STAGE0_STEPS})")
    parser.add_argument("--background", type=float, default=.3)
    parser.add_argument("--foreground", type=float, default=1.)


def base_config(args, frame_gap=None):
    from training import Config

    steps = args.steps
    if steps is None:
        steps = STAGE0_STEPS if getattr(args, "stage", None) == 0 else SWEEP_STEPS
    return Config(latent_dim=args.latent_dim, background=args.background, foreground=args.foreground,
                  steps=steps, frame_gap=frame_gap or getattr(args, "frame_gap", None) or 1)


def runtime_info(device, *, historical=False):
    import numpy
    import torch
    return {"protocol": PROTOCOL if historical else MAIN_PROTOCOL, "device": str(device),
            "torch": str(torch.__version__), "numpy": numpy.__version__}


def selection_runtime(runtime):
    # Use the versions recorded in a selected plotting runtime, including on another host.
    from gap_study import runtime_info as gap_runtime
    return {**runtime, "protocol": gap_runtime(runtime["device"])["protocol"]}


def chosen_gap(rows, runtime, override=None):
    from gap_study import selected_gap
    if override is not None:
        return override
    gap = selected_gap(rows, selection_runtime(runtime))
    if gap is None:
        raise ValueError("No frame gap passed the selection rule. See REPORT.md before choosing "
                         "an explicit --frame-gap for a separate design.")
    return gap


def main_rates(base, runtime, rows):
    from gap_study import selected_rates
    return selected_rates(rows, selection_runtime(runtime), base.frame_gap)


def main():
    from training import resolve_device
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, choices=range(4))
    group.add_argument("--calibrate-only", action="store_true", help="historical reconstruction-loss pilots")
    parser.add_argument("--shared-lr", action="store_true",
                        help="stage 1 only: all four losses at the fixed learning rate 0.001")
    parser.add_argument("--dry-run", action="store_true", help="show design and counts without training")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    add_design_arguments(parser)
    args = parser.parse_args()
    if args.shared_lr and args.stage != 1:
        parser.error("--shared-lr is defined only for stage 1")
    historical = args.stage == 0 or args.calibrate_only
    if historical and args.frame_gap not in (None, 1):
        parser.error("historical stage 0 and reconstruction-loss pilots used --frame-gap 1")
    runtime = runtime_info(resolve_device(args.device), historical=historical)
    rows = read_results()
    try:
        gap = 1 if historical else chosen_gap(rows, runtime, args.frame_gap)
        base = base_config(args, frame_gap=gap)
        if historical:
            table = calibration_table(base, runtime, rows)
            rates = select_rates(table) if table else dict.fromkeys(LOSSES, 1e-3)
        elif args.shared_lr:
            rates = dict.fromkeys(LOSSES, 1e-3)
        else:
            rates = main_rates(base, runtime, rows)
    except ValueError as exc:
        parser.error(str(exc))
    cases = calibration_configs(base) if args.calibrate_only else stage_configs(
        args.stage, base, rates, shared_lr=args.shared_lr)
    purpose = "calibration" if args.calibrate_only else "experiment"
    todo = sum(digest(identity(c, purpose, runtime)) not in rows for c in cases)
    print(f"{purpose}, stage={args.stage}, steps={base.steps}, frame_gap={gap}: "
          f"{len(cases)} unique configs, {todo} remaining")
    if args.dry_run:
        return
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUNS) + ".lock"):
        rows = read_results(RUNS, repair=True)
        if args.stage == 0 and table is None:
            print("Stage 0 timing uses provisional L2 learning rate 0.001 (no complete calibration).")
        for config in cases:
            run_one(config, purpose, runtime, args.stage, rows, RUNS)
        if args.calibrate_only:
            update_calibration_report(REPORT, base, runtime, calibration_table(base, runtime, rows))


if __name__ == "__main__":
    main()
