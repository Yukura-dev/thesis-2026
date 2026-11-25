#!/bin/bash
# Phase B of the full CUDA rerun: sanity check, stages 1-3, collapse diagnostics,
# G/H, J/K/L and experiment M. Run only after phase A selected Δ=4 and lr 0.003.
set -euo pipefail
cd ~/thesis-cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
step() { echo "=== $(date '+%F %T') START $*"; "$PY" "$@"; echo "=== $(date '+%F %T') DONE $*"; }
step run_sweep.py --filtered-sanity
step run_sweep.py --stage 1
step run_sweep.py --stage 1 --shared-lr
step run_sweep.py --stage 2
step untrained_encoder.py
step collapse_trajectories.py
step collapse_replay.py
step plot.py
step plot.py --shared-lr
step run_sweep.py --stage 3
step plot.py --stage 3
step stage3_contrast.py --source-device cuda
step switch_study.py
step texture_study.py
step final_summary.py --rescore-h-zero --device cuda
step final_summary.py
step frame_classification.py
step decoder_diagnostic.py
step initialization_study.py
step classification_summary.py --device cuda
step initialization_summary.py
step smoothing_study.py
echo "=== $(date '+%F %T') PHASE B COMPLETE"
