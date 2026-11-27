#!/usr/bin/env bash
# Reproduce every Experiment R analysis after train.py (see README.md). Outputs append to *.jsonl here.
set -euo pipefail
cd "$(dirname "$0")"
P=../../.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
phase=${1:-all}
if [[ $phase == all || $phase == g3 ]]; then
  for r in 5 8; do for s in 0 1 2 3 4; do echo "$r $s"; done; done |
    xargs -P 3 -L 1 bash -c '$0 g3.py $1 $2 > logs/g3_r$1_s$2.log 2>&1' "$P"
fi
runs() { for a in "$@"; do for r in 5 8; do for s in 0 1 2 3 4; do echo "$a $r $s"; done; done; done; }
if [[ $phase == all || $phase == traj ]]; then
  runs l1 wl1 gauss l2 wl1_ord l1_affine l1_bn l1_switch300 |
    xargs -P ${TRAJ_P:-3} -L 1 bash -c '$0 traj.py $1 $2 $3 > logs/traj_$1_r$2_s$3.log 2>&1' "$P"
fi
if [[ $phase == all || $phase == final ]]; then
  runs l1 wl1 gauss l2 wl1_ord l1_affine l1_bn l1_switch300 |
    xargs -P 4 -L 1 bash -c '$0 final.py $1 $2 $3 > logs/final_$1_r$2_s$3.log 2>&1' "$P"
fi
