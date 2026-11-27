# Experiment R (representation mechanism): pre-registered plan

Written 2026-11-27, before any R run. Codex's unpublished numbers (branch experiment/representation-mechanism, commit e2ee067, never pushed) are not used as targets.

## Questions
1. Why can position and displacement be read from z when plain L1 stalls at a flat background output?
2. Is that information acquired by training, or left over from the random initialization?
3. Why does the decoder not use it?
4. Why does weighted L1 reconstruct both circles yet give no readable velocity?
5. How do the losses differ in how z forms over training?

## Models (shared by every sub-experiment)
Thesis setup: 64x64, two frames, background 0.3, foreground 1.0, latent 16, gap 4, reflection rejection, Adam lr 0.003, batch 128, 3000 steps, clean targets, seeds 0-4, radii 5 and 8.

| arm | loss | why this setting |
|---|---|---|
| l1 | plain L1 | stalls at background (r5 5/5; r8 2/5 background, 3/5 current-only) |
| wl1 | bg_weighted_l1, w=0.1 (r5), w=0.3 (r8) | the B-3 / Experiment N settings |
| gauss | gauss_l1 sigma=0.4 | analytic smoothing; two-frame output 5/5 at both radii in Experiment M |
| l2 | L2 | reference |

**Untrained control:** the step-0 checkpoint of each run. It shares the seed's initialization.

Each run is retrained with checkpoints at steps 0 1 2 5 10 20 25 50 75 100 150 200 300 500 750 1000 1500 2000 2500 3000. The final parameter SHA-256 must match the thesis ledger. A mismatch stops the analysis of that run. Per-step training loss and the encoder and decoder update norms are logged.

Already observed before this plan (from a saved Experiment N model, r5 and r8 wl1 seed 0): each wl1 output channel draws both circles at the same brightness (out−bg about 0.64 on its own circle and on the other frame's). The decoder output therefore carries no order.

## Image metrics (fixed evaluation split 9, 1024 pairs)
- **Foreground masks:** the predicted mask is output > 0.65; the true mask is coverage > 0.5.
- **Per-frame scores:** IoU and Dice.
- **Errors:** foreground MAE and background MAE.
- **Centre and detection:** the centre is the centroid of max(out − 0.3, 0) inside the predicted mask. A frame with an empty mask is a detection failure.
- **Displacement error:** the error of the reconstructed displacement, over pairs where both frames are detected.
- **Output spatial SD:** the spatial SD of the output, per frame.

## G3: frozen encoder, decoder retrained (highest priority)
- **Encoders:** untrained (step 0), l1, wl1, gauss and l2 (step 3000). Radius 5 and 8, seeds 0-4.
- **Preprocessing of z (statistics from 8192 training pairs of split 4):**
  - none
  - center
  - scale: divide by the per-dimension SD
  - standardize
  - whiten: ZCA
  - random invertible linear: a fixed Gaussian matrix, scaled so its column norms match those of the standardization map

  Dimensions with SD below 1e-6 times the median SD get scale 1. Their count is logged. Whitening adds eps = 1e-6 times the largest eigenvalue. All maps are invertible.
- **Decoder:** a fresh copy of the thesis decoder (from_latent + deconvolutions + background offset), initialized with the same seed for every arm. It is trained for 3000 steps, Adam lr 0.003, batch 128, on the training stream of the original run.
  - Decoder losses: L1 and L2 (and gauss sigma=0.4 if L1 and L2 disagree).
- **Controls (on the l1, l2 and untrained encoders):**
  - **absorbed:** none-parametrization with the first layer initialized to W·S⁻¹, b − W·S⁻¹·m, and the reverse. This gives identical initial output to standardize, but a different parametrization.
  - **learning rate:** none with lr 0.0003 and 0.03.
  - **SGD:** none and standardize with SGD (momentum 0.9, lr 0.1).
  - **Gradients and updates:** the gradient norm per decoder layer and the realized update norm are logged at fixed steps.

Decision rules (per radius, matched decoder loss and preprocessing):
- **G3-info:** "Training under loss X added decoder-usable information" is supported only if the IoU from encoder X exceeds the untrained encoder's by ≥ 0.1 in ≥ 4/5 seeds.
- **G3-L1specific:** a preprocessing gain (standardize − none) is called L1-specific only if the median gain for the l1 encoder exceeds the l2 encoder's by ≥ 0.1.
- **G3-param:** if *absorbed* behaves like the arm whose **parametrization** it shares (not the one whose initial function it shares), preprocessing acts through optimization dynamics, not the starting point.

Prediction (stated for falsification): with an L1 decoder loss, every encoder including l2 stalls at background (IoU < 0.05) in most seeds under every preprocessing. Experiment A showed that the stall is driven by the loss's constant-magnitude background gradient, which preprocessing does not change. With an L2 decoder loss, every encoder including the untrained one reaches IoU > 0.5.

## D: trajectory of the representation
At every checkpoint, for all 40 runs:
- the image metrics above;
- latent mean and SD;
- relative parameter change of the encoder and decoder since the previous checkpoint;
- swap distance (numerator), permutation distance (denominator) and their ratio;
- ridge and MLP (thesis 64-64 probe) R² for previous position, current position, midpoint and velocity.

Probes train on split 2 (4096 pairs) and test on split 3 (2048 pairs).

- **Collapse step t_c:** the first checkpoint at which neither frame is drawn (mean centre intensity < 0.65 for both frames) and the output spatial SD is below 10 % of the clean target's spatial SD.
- **Stall interval:** from t_c to 3000, provided the run stays collapsed at every later checkpoint.
- **Cross-checkpoint transfer:** a ridge probe trained at checkpoint i is tested at checkpoint j for all pairs in the stall interval. Within-stall transfer that is much worse than same-checkpoint fits means that the representation keeps moving.

Prediction D1: for l1 r5, ≥ 80 % of the 0→3000 increase in velocity MLP R² happens before t_c. If so, "the representation keeps learning during the stall" is not claimed.

## A: layerwise probes
Run at steps 0, t_c (or 25) and 3000, for every run. The layers are the input, the four encoder blocks, z, from_latent, the three decoder blocks and the output.
- **Probes:** ridge on all features (alpha chosen on a held-out fold), ridge on PCA-16 and PCA-64, and the MLP on PCA-64.
- **Targets:** previous position, current position, midpoint, velocity, and direction (cos, sin of velocity).
- **Reference:** at the input layer, ridge should read velocity almost perfectly, because the centroid is linear in pixels when the disk mass is fixed.

## B / G5: why weighted L1 loses order
- **Decomposition:** z_sym = (E(A,B) + E(B,A))/2 and z_asym = (E(A,B) − E(B,A))/2.
- **Measures at every checkpoint:**
  - norms and variances of the two components;
  - the swap numerator and denominator in absolute terms;
  - velocity R² (ridge, MLP, big MLP) from z, z_sym and z_asym, raw and standardized.
- **Input manipulations (on rendered pairs):**
  - fix current, vary previous;
  - fix previous, vary current;
  - fix displacement, vary position;
  - fix midpoint, rotate direction;
  - swap only.

  Pairs outside the training distribution (reflection-rejected windows) are flagged.

Hypotheses:
- **W-shrink vs W-relative:** does the swap numerator fall in absolute terms (shrink), or does only the denominator grow (relative)? Decided by the step-0 → step-100 change of each. Shrink means the numerator falls by > 10×.
- **W-union (mechanism):** under wl1, drawing both circles in both channels costs only w·|fg−bg|·area for the wrong circle, so the order term in the loss is w times smaller than in L1/L2. Once the union is drawn, the remaining gradient toward order is small.
  - **Correlational check:** the order-sensitive part of the loss (loss with the true order minus loss with the frames swapped in the target) over training.
  - **Causal test, arm wl1_ord:** wl1 in which the pixels of the *other* frame's disk get weight 1 instead of w. Nothing else changes.
  - **Prediction:** wl1_ord learns order (velocity MLP R² > 0.8 and swap ratio > 0.1 in ≥ 4/5 seeds per radius). If it does not, W-union is rejected as the cause.

## G2: latent statistics (scoped by G3)
- **At step 3000 for all runs:**
  - the mean-vector norm against the input-dependent SD;
  - covariance eigenvalues, condition number and effective rank (participation ratio);
  - the norm of the decoder Jacobian dD/dz;
  - the share of the Jacobian's energy in the probe-defined position and velocity directions.
- **Further items** only if G3 shows preprocessing effects.

## E: interventions with real encodings
Directions come from real encodings, not from probes:
- z(x_swap) − z(x);
- z of a pair with the same midpoint and a rotated displacement;
- z of a pair with a shifted position;
- a random direction of equal norm.

The output response is measured as the change in reconstructed centroids per frame, together with the Mahalanobis distance of the edited z from the training z distribution.

For wl1 there is one extra test: z_sym + k·z_asym for k = 1, 3, 10, 30. If the decoder draws ordered output at some k, it can use the antisymmetric component and the order signal is only small. A null response for a background-output decoder is reported as "flat output for all z", not as "the decoder ignores the information".

## G4: normalization during L1 training
r5 and r8, seeds 0-4, plain L1:
- **fixed invertible affine:** z standardized with step-0 statistics; absorbed so the initial output is identical;
- **BatchNorm1d on z** (affine=False): a different kind of intervention;
- **switch to the fixed standardization at step 300:** absorbed, so the output is unchanged at the switch.

Prediction G4: none of these prevents the background stall (≥ 4/5 seeds collapse in each arm), because Experiment A puts the cause in the loss's background gradient. The thesis's 4.6 L2→L1 switch rescues by starting L1 from weights that already draw the circles. It changes the starting point, not the latent statistics.

## G1
Every sub-experiment above already uses r5 and r8 × 5 seeds. No seed is excluded. Every seed is reported.
