# Addendum to REPORT_ab.md (2026-11-26)

REPORT_ab.md is kept as written for the first N/O runs (commit c664065). Where the two differ, this addendum and thesis draft 18 are the current reading.

## Wording superseded in REPORT_ab.md
- Phrases such as "z holds no more than the drawn frame" or "order is lost" should be read as: the probes we tried, including the larger MLP, read no more than the matching oracle. This does not prove that z contains none of that information.
- P1 to P3 held on the 30 models they were made for. The other 15 models (background-output L1 and r=8 L1) were reported without a prediction.
- O measures first-order sums d_B and d_C over 20 Adam steps, not realised loss changes. Realised changes are much smaller (see the thesis, 4.9.2).

## Frame-swap test (PLAN_swap.md, swap_test.jsonl)
- Swap ratio R for weighted L1: at most 2.3e-4 in all 15 models (max 0.000225). For r5 L2 it is 0.56 to 0.60.
- A small R alone does not mean no order information. Background-output r8 init(a) seed 2 has R = 0.046 but swap consistency R^2 = 0.885.
- PLAN_swap.md first carried the date 2026-11-27 by mistake. It was written on 2026-11-26. The plan and its results were committed together, so git alone cannot show that the plan came first.

## Swap-augmented probe (PLAN_swap_aug.md, swap_aug.py, swap_aug.jsonl)
The velocity probe is trained on z(x) with label v together with z(x_swap) with label -v (split 2), and tested the same way (split 3).

| condition | standard MLP | big MLP |
|---|---|---|
| r5 L2 | 0.976 to 0.988 | 0.996 to 0.998 |
| r5 noisy L1 s0.4 | 0.974 to 0.986 | 0.996 to 0.998 |
| weighted L1 (15 models) | below 0.01 | below 0.07 |
| r5 L1 (background) | 0.284 to 0.823 | 0.693 to 0.925 (4/5 above 0.8) |
| r8 L1 init(a) (background) | 0.772 to 0.919 | 0.931 to 0.973 |
| one-frame models | 0.06 to 0.09 | 0.06 to 0.08 |

- Both pre-registered predictions held: weighted L1 is below 0.1 and r5 L2 is at least 0.9.
- **Current claim:** in weighted-L1 AEs, swapping the two frames hardly changes z. No probe we tried, including one trained on both orders, could read the velocity. We do not claim that order information is absent from z.
- **Side result:** background-output L1 models reach higher velocity R^2 under this probe. The success of background-output models therefore depends on the probe (thesis, threats to validity). Success in the thesis is still judged with the Section 3.4 probe.
- **Process note:** PLAN_swap_aug.md was written before the run but committed together with the results.
