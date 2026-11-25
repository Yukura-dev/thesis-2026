#!/bin/bash
# Phase A of the full CUDA rerun: diagnostics, old-spec gap study, reflections,
# then the filtered (current spec) Δ selection and learning-rate calibration.
set -euo pipefail
cd ~/thesis-cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
mkdir -p rerun_logs results_mps_archive
# Empty the sidecar ledgers once (archive the MPS versions first).
for f in collapse_reconstruction.jsonl final_h_zero_images.jsonl frame_classification_measurements.jsonl \
         initialization_trajectory.jsonl decoder_diagnostic.jsonl stage3_contrast_historical.jsonl; do
  if [ -f "results/$f" ] && [ ! -f "results_mps_archive/$f" ]; then
    cp "results/$f" "results_mps_archive/$f"
    : > "results/$f"
  fi
done
step() { echo "=== $(date '+%F %T') START $*"; "$PY" "$@"; echo "=== $(date '+%F %T') DONE $*"; }
step diagnose_steps.py
step gap_study.py --selection
step reflection_metrics.py
step reflection_study.py
step summarize_reflections.py
step filtered_geometry.py
step filtered_study.py --selection
step plot_filtered_gaps.py
step filtered_study.py --calibrate-only
echo "=== $(date '+%F %T') PHASE A COMPLETE"
