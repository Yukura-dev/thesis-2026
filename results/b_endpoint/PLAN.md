# Experiment B (endpoint study): pre-registered plan

Written before any model in this study was trained or probed (2026-11-26).

## Question
Some frozen encoders let a probe read the circle position but not its velocity. Why?
We separate three explanations:

- **H-miss (an endpoint is missing):** z encodes only one frame's position.
- **H-order (order is lost):** z encodes both positions as an unordered pair, but not which frame each belongs to.
  The magnitude and axis of the displacement are then readable, but its sign is not.
- **H-amp (error amplification):** z encodes both positions with their order.
  Small position errors are amplified when one position is subtracted from the other.
- **H-cap (probe capacity):** the information is present, but the standard MLP probe cannot extract it.

## Data and models
- Same setup as the thesis: 64x64 frames, Δ=4, reflection rejection, latent dimension 16, Adam with lr 0.003, batch 128, 3000 steps.
- Seeds 0–4. Probes train on split 2 (10,000 examples) and are evaluated on split 3 (5,000 examples).
- Every model is retrained deterministically. Its parameter SHA-256 is compared with the existing ledger (results/runs.jsonl, results/b3_weighted_l1/runs.jsonl).

| key | condition | why |
|---|---|---|
| r5_l2 | r=5, L2 | reference that succeeds |
| r5_l1 | r=5, L1 | background output |
| r5_noisy_l1_s0.1 | r=5, L1 + target noise σ=0.1 | one-frame output, velocity ≈0.07 in the thesis |
| r5_noisy_l1_s0.4 | r=5, L1 + target noise σ=0.4 | recovered |
| r8_l1 | r=8, L1 | thesis baseline |
| r8_l1_init_a | r=8, L1, initialization (a) | experiment K |
| r5_wl1_0.1 | r=5, background-weighted L1, w=0.1 | B-3: draws both circles in every frame |
| r5_wl1_0.01 | r=5, w=0.01 | B-3: "sausage" |
| r8_wl1_0.3 | r=8, w=0.3 | B-3: "sausage" |

## Measures
Each target gets its own probe on z: the existing ridge (α=1) and the existing MLP (64-64, 80 epochs).

- `prev`, `cur`: past and current centers, as one joint 4-output probe.
- `v_delta`: velocity computed as (cur − prev)/Δ from the joint endpoint probe.
- `v_direct`: velocity from a 2-output probe.
- `mid`: the midpoint (prev+cur)/2.
- `sym`: the order-invariant displacement moments (dx², dy², dx·dy).
- `v_big`, `sym_big`: the same targets with a larger MLP (256-256, 200 epochs).
- Residual slopes: per-axis regression slopes of each endpoint error on the true displacement d.
- References, computed per radius with the same probes on oracle inputs:
  - true current position only
  - true unordered pair (mid + sym)
- Reconstruction: the existing frame classification from the center intensity of each frame.

## Decision rules (fixed in advance, MLP probe)
- **ordered:** v_direct ≥ 0.8.
- **unordered (H-order):** v_direct < 0.5, sym ≥ 0.5, mid ≥ 0.9.
- **one-endpoint (H-miss):** v_direct < 0.5, sym < 0.5, max(prev, cur) ≥ 0.9.
- **partial:** anything else.
- **H-amp:** among models with prev ≥ 0.95, cur ≥ 0.95 and v_direct < 0.8, whether v_delta is less than v_direct by more than 0.1.
- **H-cap:** whether v_big − v_direct > 0.1, or sym_big − sym > 0.1. Checked for every model.

Expected residual-slope signatures:

| pattern | prev slope | cur slope |
|---|---|---|
| unordered pair → midpoint | +0.5 | −0.5 |
| current only | ≈ +1 | 0 |
| ordered | 0 | 0 |

## Predictions (stated before running)
- P1: r5_noisy_l1_s0.1 (one-frame output) is one-endpoint or unordered, not amplification.
- P2: B-3 models (w<w_crit, draw both circles or a sausage) are unordered: sym ≥ 0.5 while v_direct ≈ 0.
- P3: r5_l2 and r5_noisy_l1_s0.4 are ordered.
- r5_l1 (background output) and the r=8 L1 conditions get no directional prediction; we only report them.

## Not changed
No existing ledger row, checkpoint or weight file is modified. Outputs go to `results/b_endpoint/`. Latents and weights go to `checkpoints/b_endpoint/`.
