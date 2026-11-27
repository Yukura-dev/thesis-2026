# Experiment R follow-ups (added during the analysis)

Written 2026-11-27, after the main R runs. drift.py and cross_arm.py (F1) were run first as exploratory checks without predictions. F2 below is pre-registered here before it is run.

## Why
- G3 (r=5, 5 seeds) found that the plain-L1 encoder beats the untrained encoder by a wide margin once z is centered.
  - With an L2 decoder and standardized z: median IoU 0.63 for the L1 encoder (4/5 seeds above 0.5) and 0.46 for the untrained encoder (1/5).
- D found that z changes mostly in the first ~100 steps. After step 100, z_3000 is close to an affine image of z_100 (affine R² ≥ 0.95 in 4/5 seeds).

## F2 (g3_time.py): when is the decoder-usable information acquired?
- **Setup:** frozen plain-L1 encoders at steps 0, 5, 10, 25, 50, 100, 300, 1000 and 3000.
- **Retraining:** a fresh decoder trained with L2 under the maps none, center and standardize.
- **Everything else** is as in g3.py.
- **Runs:** r=5 and r=8, seeds 0-4.

Predictions and decision rules (r=5, standardize, median over seeds):
- **F2-transient:** the information is acquired during the collapse transient. The step-100 encoder's IoU is within 0.05 of the step-3000 encoder's, and both exceed the step-0 encoder's by ≥ 0.1.
- **F2-stall (alternative):** the information is acquired during the stall. The IoU rises by ≥ 0.1 between step 100 and step 3000.
- **Reporting:** the curve is reported per seed. Seeds that disagree with the median are not removed.
