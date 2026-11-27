#!/usr/bin/env bash
# Recompute layerwise probes for the 40 base runs after the ridge fix (centre-only, float64).
cd "$(dirname "$0")"
while pgrep -f 'bin/python g3' >/dev/null; do sleep 30; done
[ -f layers.jsonl ] && mv layers.jsonl layers_superseded_standardized_ridge.jsonl
export OMP_NUM_THREADS=2
run() { for a in "$@"; do for r in 5 8; do for s in 0 1 2 3 4; do
  ../../.venv/bin/python traj.py $a $r $s > logs/layers_${a}_r${r}_s${s}.log 2>&1
done; done; done; }
run l1 wl1 & run gauss l2 & wait
wc -l layers.jsonl
