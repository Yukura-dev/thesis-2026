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
SEEDS = (0, 1, 2)
PILOT_SEED = 1729
RATES = (3e-4, 1e-3, 3e-3)
PROTOCOL = "circle-ae-v1"


def canonical(config):
    if config.noise_level == 0:
        return replace(config, noise_type="gaussian", placement="target_only", noise_level=0.)
    return config


def stage_configs(stage, base, rates):
    """Return unique configs in execution order; zero noise has one identity."""
    if stage == 0:
        return [replace(base, radius=5, lr=rates["l2"], loss="l2", seed=0)]
    if stage == 1:
        cells = [(r, "gaussian", 0., "target_only", ("l1", "l2")) for r in RADII]
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
    configs = (canonical(replace(base, radius=r, noise_type=kind, noise_level=level,
                                 placement=placement, loss=loss, lr=rates[loss], seed=seed))
               for r, kind, level, placement, losses in cells for loss, seed in product(losses, SEEDS))
    return list(dict.fromkeys(configs))


def calibration_configs(base):
    return [replace(base, radius=5, noise_type="gaussian", noise_level=0., placement="target_only",
                    loss=loss, seed=PILOT_SEED, lr=lr) for loss, lr in product(LOSSES, RATES)]


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def identity(config, purpose, runtime):
    return {"config": asdict(canonical(config)), "purpose": purpose, "runtime": runtime}


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
    outcome = run_experiment(config, device=runtime["device"], calibration=(purpose == "calibration"))
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
    parser.add_argument("--latent-dim", type=int, choices=(4, 16, 64), default=16)
    parser.add_argument("--steps", type=int, default=None, help="default: training.Config.steps")
    parser.add_argument("--background", type=float, default=.3)
    parser.add_argument("--foreground", type=float, default=1.)


def base_config(args):
    from training import Config

    kw = {} if args.steps is None else {"steps": args.steps}
    return Config(latent_dim=args.latent_dim, background=args.background, foreground=args.foreground, **kw)


def main():
    import numpy
    import torch
    from training import resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, choices=range(4))
    group.add_argument("--calibrate-only", action="store_true", help="measure three rates for each of four losses")
    parser.add_argument("--dry-run", action="store_true", help="show design and counts without running training")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    add_design_arguments(parser)
    args = parser.parse_args()
    base = base_config(args)
    runtime = {"protocol": PROTOCOL, "device": str(resolve_device(args.device)),
               "torch": str(torch.__version__), "numpy": numpy.__version__}
    default_rates = dict.fromkeys(LOSSES, 1e-3)
    if args.dry_run:
        rows = read_results()
        table = calibration_table(base, runtime, rows)
        pilots = calibration_configs(base)
        missing = sum(digest(identity(c, "calibration", runtime)) not in rows for c in pilots)
        if args.calibrate_only or args.stage in (1, 2, 3):
            print(f"LR calibration: {len(pilots)} trials, {missing} remaining")
        if not args.calibrate_only:
            cases = stage_configs(args.stage, base, select_rates(table) if table else default_rates)
            print(f"stage {args.stage}: {len(cases)} unique configs")
            if table or args.stage == 0:
                todo = sum(digest(identity(c, "experiment", runtime)) not in rows for c in cases)
                print(f"{todo} remaining")
        return
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUNS) + ".lock"):
        rows = read_results(RUNS, repair=True)
        if args.calibrate_only or args.stage in (1, 2, 3):
            for config in calibration_configs(base):
                run_one(config, "calibration", runtime, None, rows, RUNS)
        table = calibration_table(base, runtime, rows)
        if table:
            update_calibration_report(REPORT, base, runtime, table)
        if args.calibrate_only:
            return
        rates = select_rates(table) if table else default_rates
        if table is None:
            print("Stage 0 timing uses the provisional L2 learning rate 0.001 (no complete calibration).")
        for config in stage_configs(args.stage, base, rates):
            run_one(config, "experiment", runtime, args.stage, rows, RUNS)


if __name__ == "__main__":
    main()
