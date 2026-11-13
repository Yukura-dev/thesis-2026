"""Estimate the full stage 1→2→3 design on the measured machine, without training."""
import argparse
from itertools import product
import json
from math import ceil
from statistics import median

from gap_study import CALIBRATION, SELECTION, STEPS, runtime_info, selected_gap, selected_rates
from run_sweep import LOSSES, ROOT, read_results, stage_configs
from training import Config


def estimate(rows, generation):
    stage0 = next(r for r in rows.values() if r["purpose"] == "experiment" and r["first_stage"] == 0)
    machine = {k: stage0["runtime"][k] for k in ("device", "torch", "numpy")}
    measured = [r for r in rows.values() if all(r["runtime"].get(k) == v for k, v in machine.items())]
    runtime = {**runtime_info(machine["device"]), **machine}
    gap, rates = None, None
    try:
        gap = selected_gap(rows, runtime)
        status = "no_qualifying_gap" if gap is None else "gap_selected"
    except ValueError:
        status = "gap_selection_incomplete"
    if gap is not None:
        try:
            rates = selected_rates(rows, runtime, gap)
        except ValueError:
            status = "lr_calibration_incomplete"

    historical = [r for r in measured if r["purpose"] == "calibration"
                  and r["config"]["steps"] == STEPS and r["config"]["radius"] == 5
                  and r["config"]["noise_level"] == 0 and r["config"].get("frame_gap", 1) == 1]
    pilots = [r for r in measured if r["purpose"] == CALIBRATION and r["runtime"] == runtime
              and r["config"]["frame_gap"] == gap and r["config"]["steps"] == STEPS]
    reference_gap = 1 if gap is None else gap
    gap_times = [r["timing"]["train_seconds"] for r in measured
                 if r["purpose"] == SELECTION and r["runtime"] == runtime
                 and r["config"]["frame_gap"] == reference_gap and r["config"]["radius"] == 5
                 and "reused_from" not in r]
    old_l2 = [r["timing"]["train_seconds"] for r in historical if r["config"]["loss"] == "l2"]
    offset = median(gap_times) - median(old_l2) if gap_times else 0.
    clean, shared_train = {}, {}
    for loss in LOSSES:
        candidates = [r for r in pilots if r["config"]["loss"] == loss]
        fixed = [r for r in candidates if r["config"]["lr"] == 1e-3]
        fixed_old = [r for r in historical if r["config"]["loss"] == loss and r["config"]["lr"] == 1e-3]
        shared_train[loss] = (median(r["timing"]["train_seconds"] for r in fixed) if fixed else
                              median(r["timing"]["train_seconds"] for r in fixed_old) + offset)
        if rates is not None:
            candidates = [r for r in candidates if r["config"]["lr"] == rates[loss]]
        source = candidates or [r for r in historical if r["config"]["loss"] == loss]
        sec = median(r["timing"]["train_seconds"] for r in source)
        clean[loss] = {"train_seconds": sec if candidates else sec + offset,
                       "source": "gap_calibration" if candidates else "historical_calibration",
                       "measurements": len(source)}
    # Main stages run one joint 6-output probe per method, without the diagnostic
    # 4-output endpoint fits or extra pixel-change accumulator. Use those actual
    # fit+evaluation times. New splits have exactly the same feature/target shape.
    regular = [r for r in measured if r["purpose"] in ("experiment", "sanity", "velocity_sanity")
               and r["config"]["probe_test"] == 5000 and r["config"]["probe_train"] == 10000
               and r["config"]["noise_level"] == 0 and "temporal_reconstruction" not in r]
    evaluation = median(r["timing"]["evaluation_seconds"] for r in regular)
    overhead_sources = pilots or historical
    overhead = median(r["timing"]["total_seconds"] - r["timing"]["evaluation_seconds"]
                      - r["timing"]["train_seconds"] for r in overhead_sources)
    gen = {(r["radius"], r["noise_type"], r["noise_level"], r["placement"]):
           median(r["seconds_per_batch"]) for r in generation["cells"]}
    reference = gen[(5, "gaussian", 0., "target_only")]
    base = Config(frame_gap=reference_gap, steps=STEPS)
    eval_batches = ceil(base.probe_train/base.batch_size) + ceil(base.probe_test/base.batch_size)
    validation_batches = ceil(1024/base.batch_size)
    header = {"r5_clean": clean, "historical_to_gap_train_offset_s": offset,
              "gap_r5_l2_training_anchors": len(gap_times),
              "normal_probe_evaluation_s": evaluation, "normal_evaluation_rows": len(regular),
              "non_training_non_probe_s": overhead, "gap_runtime_reference": reference_gap, "shared_lr_train_s": shared_train}

    def cost(cases, shared=False):
        core_s = add_s = 0.
        for c in cases:
            train_s = shared_train[c.loss] if shared else clean[c.loss]["train_seconds"]
            core_s += train_s + evaluation + overhead
            # Change only the CPU generation cost relative to the clean anchor.
            train_g = gen[(c.radius, c.noise_type, c.noise_level, c.placement)]
            if c.noise_level > 0 and c.placement in ("input_only", "both"):
                eval_g = gen[(c.radius, c.noise_type, c.noise_level, "input_only")]
            else:
                eval_g = gen[(c.radius, "gaussian", 0., "target_only")]
            add_s += STEPS * max(0., train_g - reference)
            add_s += (eval_batches + validation_batches) * max(0., eval_g - reference)
        return {"runs": len(cases), "overlapped_hours": core_s/3600,
                "serialized_hours": (core_s + add_s)/3600}

    # Counts and timings for the full planned design, deduplicated between stages;
    # only main stage rows would have final probe IDs 6/7. Never reuse pilots.
    planned_rates = rates or dict.fromkeys(LOSSES, 1e-3)
    seen, stages = set(), []
    for stage in (1, 2, 3):
        cases = set(stage_configs(stage, base, planned_rates))
        stages.append({"stage": stage, **cost(sorted(cases-seen, key=repr))})
        seen.update(cases)
    main = {k: sum(s[k] for s in stages) for k in ("runs", "overlapped_hours", "serialized_hours")}
    optional = set(stage_configs(1, base, planned_rates, shared_lr=True))
    if rates is None:
        # One matching primary LR saves 5 radii × 5 seeds for that loss.
        extras = []
        for lr_l1, lr_l2 in product((1e-3, 3e-3), repeat=2):
            scenario = {**planned_rates, "l1": lr_l1, "l2": lr_l2}
            extra = optional - set(stage_configs(1, base, scenario))
            extras.append(cost(sorted(extra, key=repr), shared=True))
    else:
        extras = [cost(sorted(optional - seen, key=repr), shared=True)]
    sensitivity = {"min_runs": min(s["runs"] for s in extras),
                   "max_runs": max(s["runs"] for s in extras),
                   "overlapped_hours": min(s["overlapped_hours"] for s in extras),
                   "serialized_hours": max(s["serialized_hours"] for s in extras)}
    return {"status": status, "selected_gap": gap, "selected_lr": rates,
            "runtime": machine, "anchors": header, "stages": stages, "main": main,
            "shared_lr_extra": sensitivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print full estimates and timing anchors")
    args = parser.parse_args()
    result = estimate(read_results(), json.loads((ROOT/"results"/"generation_times.json").read_text()))
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(f"Status: {result['status']}; selected gap: {result['selected_gap']}")
    print("Serial runs | clean-fit estimate if added generation overlaps | if added generation is serial")
    for name, r in [(f"Stage {s['stage']}", s) for s in result["stages"]] + [("Total", result["main"])]:
        print(f"{name}: {r['runs']} runs, {r['overlapped_hours']:.2f}–{r['serialized_hours']:.2f} h")
    s = result["shared_lr_extra"]
    print(f"Shared LR (optional, additional): {s['min_runs']}–{s['max_runs']} runs, "
          f"{s['overlapped_hours']:.2f}–{s['serialized_hours']:.2f} h")
    print("Estimates exclude calibration, selection and plotting. Before a gap is chosen, timing "
          "is conditional on historical clean fits. Scenarios do not bound observed wall time.")


if __name__ == "__main__":
    main()
