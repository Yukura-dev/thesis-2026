# Experiment A (gradient study): pre-registered plan

Written after Experiment B and before any A run (2026-11-26).

## What B changed
L1 models with background output still encode a good deal of position and partial order in z: a larger MLP reads velocity at R² 0.74 (r=5).
So the encoder does learn. The stall sits on the output side: the decoder does not brighten the circle.
A therefore asks one question: **during real Adam training, how do the background and object pixel groups pull on the update?**
We do not add conditions beyond those needed to answer it.

## Conditions
Radius r=5 only, seeds 0–4, the thesis training setup (Δ=4, reflection rejection, Adam lr 0.003, 3000 steps, clean targets):

- **L1:** stalls at background output in 5/5 seeds.
- **gauss_l1 with σ=0.05:** the smallest smoothing that leaves background output in the thesis (one-frame output).
  It uses clean targets, so the pixel groups are defined identically in both arms.

Each run is checked against the thesis ledger SHA-256. This shows that the measurement does not perturb training.

## Pixel groups (fixed 512-example evaluation batch, split 9)
- **B:** clean pixel equals the background value.
- **E:** partially covered (anti-aliased boundary).
- **K:** fully covered core.

C = E ∪ K. Each group loss is summed and divided by the total pixel count, so L = L_B + L_C and ∇L = ∇L_B + ∇L_C.
Additivity is asserted at every measurement.

## Measures
- Checkpoints at steps 0, 50, 300, 1000 and 2980.
  - Group gradient norms and cos(∇L_B, ∇L_C).
  - Over the next 20 real Adam steps, the accumulated first-order changes d_B = Σ ∇L_B·u and d_C = Σ ∇L_C·u, where u is the actual parameter update.
  - The realised changes of L_B and L_C over the same 20 steps.
- **Counterfactual:** at the same weights, the group gradients under the other loss (L1 ↔ gauss_l1 with σ=0.05).
- A trace every 25 steps: L_B, L_E, L_K, mean output on core pixels, background MAE.

## Hypotheses and decision rules
- **HA-cancel:** in L1 at steps ≥ 300, the Adam update lowers the object loss (d_C < 0), but raises the background loss by a comparable amount. That is, ρ = −d_B/d_C lies in [0.5, 2] in the median over seeds and stall checkpoints.
- **HA-dominance (alternative):** in L1 the update hardly touches the object loss: |d_C| < 0.1·|d_B| in the median.
- **HA-smooth:** at the same weights, gauss_l1's background gradient norm is less than 0.2× that of L1 (median), while the object gradient norm stays within 0.5–2× of L1's.
  This would show that smoothing acts by removing the background term's first-order pull.
- If neither HA-cancel nor HA-dominance holds, we report the observed pattern without claiming a mechanism.
