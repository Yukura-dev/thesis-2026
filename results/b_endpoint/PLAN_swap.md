# Experiment N addendum: frame-swap test and past-only oracle (pre-registered)

Written 2026-11-27, after an external review and before running anything below.

## Why
"z matches the oracle" shows only that the probes we tried read no more than the oracle does.
The frame-swap test asks the encoder directly whether it distinguishes the two frame orders.

## Frame-swap test
- Uses the saved 45 models (weights must match the ledger SHA). No retraining.
- Test split 3 (5,000 examples). x_swap is the same pair with the frames exchanged, which is the reversed movie with velocity −v.
- **Swap ratio:** R = mean‖z(x) − z(x_swap)‖² / mean‖z(x) − z(x')‖², where x' is a different test example (fixed random permutation).
  - R ≈ 0 means z ignores the frame order.
- **Swap consistency:** the 2-output velocity MLP is fitted on split 2 as before, then applied to z(x_swap). Its R² is scored against −v.

## Past-only oracle
The same MLP probes, given the true past position only. This is the reference for "past only" models, the mirror of the existing current-only oracle.

## Predictions
- Weighted-L1 models (all 15): R is at most 1/10 of the mean R of the r5_l2 models, and the swap-consistency R² is below 0.1.
- r5_l2 and r5 noisy σ=0.4 (10): swap-consistency R² ≥ 0.8.
- In past-only models, the R² of the undrawn current position is within 0.02 of the past-only oracle.
- Background and one-frame conditions: reported without prediction.
