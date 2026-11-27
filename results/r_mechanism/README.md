# Experiment R: how the representation forms (L1 stall, weighted L1, smoothing)

PLAN.md was committed (f1b76b7) before any run. REPORT.md is the write-up; tables.md and figures/ are generated.

## Reproduce (WSL, ~/thesis-cuda, CUDA)
```bash
cd results/r_mechanism
P=../../.venv/bin/python
$P train.py l1 wl1 gauss l2 wl1_ord l1_affine l1_bn l1_switch300   # 80 runs, checkpoints + step logs
for r in 5 8; do for s in 0 1 2 3 4; do G3_FAST=0 $P g3.py $r $s; done; done   # G3, ~25 min per pair
for a in l1 wl1 gauss l2 wl1_ord l1_affine l1_bn l1_switch300; do for r in 5 8; do for s in 0 1 2 3 4; do
  $P traj.py $a $r $s; $P final.py $a $r $s; done; done; done                     # D, A, B/G5, G2, E
$P drift.py; $P cross_arm.py                                                      # follow-ups
$P summarize.py                                                                   # tables.md, figures/
```
Each script appends to its jsonl; delete the jsonl first to start over. traj.py skips runs already present.
WSL had 7.7 GB of RAM: run at most two analysis processes at once (each holds ~2.5 GB, mostly CUDA libraries).

## Files
| file | content |
|---|---|
| mech.py | shared code: arms, training loop with checkpoints, data, image metrics, probes |
| train.py, train.jsonl | training; base arms are checked against the thesis ledger (40/40 SHA-256 match) |
| g3.py, g3.jsonl | frozen encoder, retrained decoders (108 per radius-seed pair) |
| traj.py, traj.jsonl, layers.jsonl | per-checkpoint image metrics, probes, swap test; layerwise probes |
| final.py, g2.jsonl, interventions.jsonl | latent and decoder-entry statistics; interventions with real encodings |
| drift.py, drift.jsonl | representation drift between checkpoints (CKA, affine R²) |
| cross_arm.py, cross_arm.jsonl | the same seed under different losses, same step |
| summarize.py, tables.md, figures/ | tables and figures |
| logs/ | per-job logs |

## Checkpoints
Not in git (`checkpoints/` is ignored). They live in `~/thesis-cuda/checkpoints/r_mechanism/<arm>_r<radius>_s<seed>/<step>.pt`
(80 runs × 20 checkpoints, about 0.25 MB each, plus `steplog.npy` with per-step loss, swapped-target loss and
encoder/decoder update norms). They are regenerated bit-for-bit by train.py (base arms match the ledger SHA-256).
