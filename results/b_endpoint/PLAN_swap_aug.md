# Experiment N addendum 2: swap-augmented velocity probe (pre-registered)

Written 2026-11-26, after the second external review and before running anything below.

## Why
In the frame-swap test, the velocity probe was fitted on z(x) only. A probe that cannot read velocity from z(x) at all would also fail on z(x_swap), so a negative swap-consistency R² does not by itself show that z holds no order information.
This test gives the probe both orders explicitly, so that any order signal left in the small differences z(x) − z(x_swap) can be used.

## Method
- Same 45 saved models (weights must match the ledger SHA), no retraining.
- Probe training set: split 2 (10,000 movies), used twice: z(x) with label v, and z(x_swap) with label −v (20,000 rows).
- Test set: split 3 (5,000 movies), likewise both orders, scored against v and −v (10,000 rows). R² is the mean over the two velocity axes.
- Probes: the standard 2-output MLP probe (predict_probes) and the larger 256-256 MLP (fit_big), both on inputs standardized with training-set moments.

## Predictions
- Weighted-L1 models (all 15): augmented-probe velocity R² below 0.1 with both probes.
- r5_l2 models: augmented-probe velocity R² at least 0.9 with the standard probe (positive control).
- The other conditions are reported without a prediction.

## Reading
If the weighted-L1 prediction holds, the thesis may say that even a probe trained on both orders could not read the velocity. It still may not say that the order information is absent from z.
