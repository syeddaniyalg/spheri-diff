# Experiment Log

This page records the pilot experiment in the order it happened. Each stage states what was asked, what was done, what was found, and why the next stage followed. It is written so that a reader (or an assistant) can take up the work without the earlier conversations.

## Contents

1. [Goal and fixed rules](#1-goal-and-fixed-rules)
2. [Stage 1: first pilot at L = 16](#2-stage-1-first-pilot-at-l--16)
3. [Stage 2: the band-limit floor and L = 31](#3-stage-2-the-band-limit-floor-and-l--31)
4. [Stage 3: first regional fine-tuning](#4-stage-3-first-regional-fine-tuning)
5. [Stage 4: pooled levels and 16-bit spectral stages](#5-stage-4-pooled-levels-and-16-bit-spectral-stages)
6. [Stage 5: targeted changes to the pooled model](#6-stage-5-targeted-changes-to-the-pooled-model)
7. [Stage 6: local processing (last bet)](#7-stage-6-local-processing-last-bet), then [Stage 7: per-degree channel mixing](#7b-stage-7-per-degree-channel-mixing-in-progress)
8. [Summary table](#8-summary-table)
9. [What is understood now](#9-what-is-understood-now)
10. [Open questions and possible next steps](#10-open-questions-and-possible-next-steps)
11. [Where things are](#11-where-things-are)

## 1. Goal and fixed rules

- **Goal.** Match or beat WIND-S with fewer parameters and a similar training time, and show better transfer to a regional patch at a finer resolution through Slepian fine-tuning.
- **Baseline.** Our re-implementation of WIND in three sizes: S (3,005,369 parameters), M (7,336,121) and L (12,238,201). WIND-S is the reference to beat.
- **Data.**
  - Global: ERA5 from WeatherBench 2 on a 32 x 64 equiangular grid with poles (about 5.6 degrees), every 6 hours. Train 2010 to 2019, validation 2020, test 2021.
  - 25 predicted channels: geopotential, temperature, specific humidity, u and v wind at 250, 500, 700 and 850 hPa (20), plus 2 m temperature, mean sea level pressure, 10 m u and v wind, and 6-hour precipitation (5). Three static fields (land-sea mask, soil type, surface geopotential) are inputs only.
  - Regional patch: 16 x 16 points at 1.5 degrees over Pakistan and northern India (18.0 to 40.5 N, 58.5 to 81.0 E). Train 2017 to 2019, same validation and test years, normalised with the global statistics.
- **Budget.** 25,000 pre-training steps and 5,000 fine-tuning steps for every model. The first pilot used 45,000 steps; the budget was reduced to 25,000 afterwards, and WIND values at 25,000 steps are interpolated from its 45,000-step runs.
- **Measures.**
  - Validation nRMSE: area-weighted RMSE of the 3-member ensemble mean on standardised fields, averaged over the 25 channels and the 4 generated frames, on 32 windows of 2020 with 15 DDIM steps. Lower is better.
  - Final protocol `final_v1`: 64 windows of 2021, 5 members, a 20-frame forecast and six reconstruction tasks solved with MMPS.
- **Rules.** Seed 42 for everything. Hardware: two T4 GPUs on Kaggle for full runs, an RTX 4050 laptop GPU for short local tests.

## 2. Stage 1: first pilot at L = 16

- **What was done.** WIND S, M and L and SpherIDiff `b4` (1,601,561 parameters) were trained for 45,000 steps. SpherIDiff used the band limit of the original proposal, L = 16 (K = 256), with a dense spectral weight per block, a channel MLP and a temporal attention in each of 4 blocks. Smaller sizes `xs`, `s` and `m` were also defined.
- **Result.**
  - At 25,000 steps: SpherIDiff b4 0.312, WIND-S 0.260, WIND-M 0.241, WIND-L 0.244.
  - SpherIDiff had about half the parameters of WIND-S but was much slower: 6.4 h against 1.4 h for 45,000 steps.
- **Why the next stage.** SpherIDiff learned fast early but levelled off above WIND. The question became whether the band limit itself made this error unavoidable.

## 3. Stage 2: the band-limit floor and L = 31

- **The floor.** The error between the data and its band-limited version, which no model at that band limit can go below. At L = 16 it is 0.274, already above WIND-S (0.260). So the L = 16 model could not reach WIND-S at all.
- **Why L = 16.** The proposal took L from the sampling rule of the equiangular grid, L = min(floor((H + 1)/2), floor(W/2) + 1), which gives 16 on 32 x 64.
- **The change.** The analysis was fitted by weighted least squares instead: choose the coefficients so that synthesis after analysis reproduces the data as closely as possible, with area weights. This works as long as the grid can tell all harmonics apart (full column rank), which holds up to L = min(H - 1, W/2) = 31 and fails at 32. The floor falls to 0.101 at L = 31.
- **Other changes for this stage.**
  - The weight became diagonal plus low rank, W = D + P Q^T with rank 32, to keep the parameters low (a dense 961 x 961 weight has 923,521 entries).
  - The blocks ran on an internal Gauss grid. The paper does not mention this.
- **Result (25,000 steps).**

  | Model | Parameters | Time | Validation nRMSE |
  |---|---|---|---|
  | L = 31 xs | 264,987 | 1.62 h | 0.366 |
  | L = 31 s | 335,901 | 2.25 h | 0.367 |
  | L = 31 m | 587,805 | 3.52 h | 0.319 |
  | L = 31 b4 | 1,589,277 | 5.88 h | 0.289 |

- **Reading.** L = 31 improved b4 from 0.312 to 0.289, and larger models helped. But b4 took 5.88 h, about 7.5 times WIND-S. A rank-128 ablation was set up; its results are not in the repository.
- **Why the next stage.** Before fixing speed, test the second goal: transfer to the patch.

## 4. Stage 3: first regional fine-tuning

- **Method.**
  - WIND: all weights are fine-tuned on the patch.
  - SpherIDiff: the spectral stage works in the Slepian basis of the patch. The pre-trained weight is restricted to the M best-concentrated Slepian functions, G0 = U_M^T W U_M, and frozen. A trainable M x M update dG is added, starting at zero.
  - Two adaptations were compared: `delta_and_pointwise` (dG plus the pointwise layers) and `delta_only` (dG alone).
  - The regional band limit was L' = 61, which gives M = 45.
- **Result (5,000 steps).**
  - WIND-S 0.295, WIND-M 0.279, WIND-L 0.283.
  - SpherIDiff L = 16 b4 0.358 and L = 31 b4 0.358.
  - `delta_only` was much worse (0.583 and 0.605), so the pointwise layers must be fine-tuned too.
  - Zero-shot (no training on the patch): SpherIDiff 0.515 (L = 16) and 0.815 (L = 31), against 1.233 for WIND-S.
- **Reading.** SpherIDiff transfers much better before fine-tuning, but WIND is better after it. SpherIDiff fine-tuning was also slower.
- **Why the next stage.** Training time was the main obstacle, so the next step was to reduce it without changing the mathematics.

## 5. Stage 4: pooled levels and 16-bit spectral stages

- **Idea.** Run the blocks at several resolutions, like a U-Net. Each level halves the grid, and its band limit follows the least-squares rule: L = 31, 15, 7 (and 3 for a fourth level). Pooling keeps the degrees below the coarser band limit; unpooling synthesises on the finer grid and adds the skip. Widths 64, 128, 256. The spectral stages and pooling run in 16-bit precision, which cut the time per step by about 25% in a local test without changing the error.
- **Regional version.** Every level uses the Slepian functions of the patch at its own band limit: L' = 120 (the least-squares limit of the 1.5-degree grid), then half at each coarser level, giving M = 175, 44 and 11. Pooling on the patch is a least-squares fit, because Slepian bases at different band limits are not nested. The regional floor falls from 0.208 at L' = 61 to 0.073 at L' = 120.
- **Variants.** 3 or 4 levels, each with a dense weight or with rank 32.
- **Result.**

  | Variant | Parameters | Time | Global | Regional |
  |---|---|---|---|---|
  | 3 levels, dense | 2,658,238 | 1.24 h | 0.340 | 0.340 |
  | 3 levels, rank 32 | 864,910 | 1.21 h | 0.344 | 0.336 |
  | 4 levels, dense | 2,674,448 | 0.91 h | 0.369 | 0.361 |
  | 4 levels, rank 32 | 882,408 | 0.91 h | 0.368 | 0.360 |

- **Reading.**
  - Training time fell from 5.88 h to 1.24 h, about 1.6 times WIND-S, but accuracy dropped (0.340 against 0.289).
  - Rank 32 matched dense in validation but gave much worse 20-frame forecasts (0.765 against 0.555), so dense was kept.
  - 4 levels was faster but less accurate.
  - **Chosen base: 3 levels, dense, 16-bit spectral stages.**
- **Why the next stage.** The floor was no longer the limit (0.10 globally), so the remaining error came from learning within the band. Small targeted changes were tested first.

## 6. Stage 5: targeted changes to the pooled model

- **Changes, each tested alone on the base.**
  - **Degree filters:** each channel gets its own diagonal correction per degree, W_c = W + diag(kappa_c(l)), starting at zero. Reason: one shared W gives every channel the same spatial filter.
  - **Noise conditioning:** the network receives each frame's noise level through a small embedding. Reason: a spectral operator followed by pointwise layers cannot easily estimate the noise level of a whole frame. This departs from WIND, which infers the level.
  - **Space-time attention:** at the coarsest level, one attention over all points and frames replaces the temporal attention.
- **Result.**

  | Variant | Global | Regional |
  |---|---|---|
  | base | 0.340 | 0.340 |
  | + degree filters | 0.324 | 0.340 |
  | + noise conditioning | 0.330 | 0.333 |
  | + space-time attention | 0.334 | 0.339 |

- **Follow-up.** The filters were extended to one value per coefficient, W_c = W + diag(kappa_c(l, m)), and combined with noise conditioning: 2,925,310 parameters, 1.12 h, global 0.323, regional 0.334, zero-shot 0.731.
- **Reading.** Each change gave 0.006 to 0.016 globally, and the gains mostly did not carry over to fine-tuning. Single-seed noise is about 0.008 (measured at 1,000 steps), so the small ones are not conclusive.
- **Why the next stage.** Global spectral operators learn the large, dominant scales first and the fine scales slowly. That pointed to the lack of local processing.

## 7. Stage 6: local processing (last bet)

Two variants were run in parallel, one per T4. Both keep the spectral stages, spectral pooling and noise conditioning.

- **Pooled + spherical convolution** (DISCO-type, own implementation, after Ocampo et al. 2023).
  - Built on the Stage 5 model with coefficient filters and noise conditioning.
  - A local convolution on the sphere at the end of the two blocks of the finest level.
  - Four fixed basis functions within about 5.8 degrees (180/31): a central bump, a ring, and the ring times the cosine and sine of the direction from local north. They capture a local average, a ring average and the two horizontal gradients.
  - Each basis function is normalised by its continuous integral, so the same weights mean the same physical operator on the global grid and on the patch.
  - It is computed as a sparse matrix product in 16-bit precision, starts at zero, and adds 32,896 parameters.
  - The torch-harmonics version was too slow (about 742 ms per layer), failed in fp16, and assumed a full global grid, so the operator was rebuilt as a sparse matrix (about 12 to 18 ms) and checked against torch-harmonics to 1e-6.
- **Hybrid U-ViT.**
  - Per level: a spectral mixer (normalisation and the spectral stage, with per-degree filters), then a WIND residual block (two 3 x 3 convolutions with group norm and noise-level FiLM) at levels 0 and 1.
  - At level 2: the spectral mixer and one Transformer block attending over all 640 frame-point tokens.
  - Spectral weights are shared by the down and up paths of a level.
  - There is no channel MLP and no temporal attention, and the convolutions use zero padding.
  - About 60% of the parameters are in the convolution and Transformer blocks.
- **Result.**

  | Variant | Parameters | Time | Global | Regional | Zero-shot |
  |---|---|---|---|---|---|
  | Pooled + spherical convolution | 2,958,206 | 1.48 h | **0.305** | **0.324** | 0.764 |
  | Hybrid U-ViT | 2,685,308 | 0.90 h | 0.370 | 0.329 | 0.965 |
  | WIND-S (reference) | 3,005,369 | 0.78 h | 0.260 | 0.295 | 1.233 |

- **Final protocol (global, test 2021).**
  - The spherical convolution improved every task over the model without it, most clearly spatial reconstruction (0.548 to 0.506) and spatiotemporal reconstruction (0.437 to 0.398).
  - The hybrid matched WIND-S on spatial reconstruction (0.454 against 0.457; regionally 0.340 against 0.340) but was clearly worse on temporal reconstruction (0.398 against 0.224).
- **Reading.**
  - Local processing gave the largest single gain.
  - The hybrid's pattern (good in space, weak in time) fits its design, in which frames meet only in one coarse Transformer. But the hybrid differs from the pooled model in several ways at once, so the cause is not isolated.
  - The remaining gap to WIND-S is 0.045 globally and 0.029 regionally.

## 7b. Stage 7: per-degree channel mixing (in progress)

- **Reason.** The dense spectral weight held 73% of the parameters of the pooled model, mixed coefficients but never channels, and a rank-32 form reached almost the same validation error. The hypothesis is that these parameters do more good as scale-dependent channel mixing.
- **Decision (2026-11-28).** The spectral stage changes permanently to f_lm A(l) + (m/L) f_l,-m B(l), with A and B interpolated from 8 knots in the degree. The dense weight and the filters are removed; no learned mode mixing is used in pre-training. The spherical convolution and noise conditioning stay. Fine-tuning freezes the pre-trained knots, adds new knots (zero) above the pre-trained band limit and keeps the trainable dG; the pointwise layers and the spherical convolution are also fine-tuned. Details: [models.md](models.md#per-degree-channel-mixing).
- **Local test (1,000 steps, RTX 4050, same settings as the earlier 1,000-step tests).** Validation nRMSE 0.469 against 0.535 for the current best (`pooled3_disco`), at 0.191 against 0.199 s per step, with 2,415,897 parameters against 2,958,206.
- **Next.** The full run of 25,000 + 5,000 steps with `notebooks/kaggle_pilot_channel_mixing.ipynb`.

## 8. Summary table

| Model | Params | Time to 25k (h) | s/step | Global nRMSE | Regional nRMSE | Zero-shot |
|---|---|---|---|---|---|---|
| WIND-S | 3,005,369 | 0.78 | 0.113 | 0.260 | 0.295 | 1.233 |
| WIND-M | 7,336,121 | 0.91 | 0.131 | 0.241 | 0.279 | 1.204 |
| SpherIDiff L = 16 b4 | 1,601,561 | 3.55 | 0.511 | 0.312 | 0.358 | 0.515 |
| SpherIDiff L = 31 b4 (rank 32) | 1,589,277 | 5.88 | 0.849 | 0.289 | 0.358 | 0.815 |
| Pooled (3 levels, dense, fp16) | 2,658,238 | 1.24 | 0.179 | 0.340 | 0.340 | 0.828 |
| + coefficient filters + noise level | 2,925,310 | 1.12 | 0.162 | 0.323 | 0.334 | 0.731 |
| + spherical convolution | 2,958,206 | 1.48 | 0.212 | 0.305 | 0.324 | 0.764 |
| Hybrid U-ViT | 2,685,308 | 0.90 | 0.130 | 0.370 | 0.329 | 0.965 |

Global: validation after 25,000 steps. Regional: validation after 5,000 fine-tuning steps. Zero-shot: regional validation before fine-tuning. The L = 16 and L = 31 rows were fine-tuned with L' = 61, the others with L' = 120.

## 9. What is understood now

1. **The band-limit floor is solved.** Least squares raised L from 16 to 31 and the floor from 0.274 to 0.101, well below WIND.
2. **Speed is close to acceptable.** Pooling and 16-bit spectral stages cut training from 7.5 times WIND-S to about 1.4 to 1.9 times.
3. **Zero-shot transfer is a clear strength.** Every SpherIDiff model beats WIND before fine-tuning, because its operators are defined on the sphere and not on grid pixels.
4. **The gap after training is in learning within the band.** Global spectral operators learn large scales quickly and fine scales slowly. Local processing helped most.
5. **Time coupling matters.** Temporal attention at every level is needed for the temporal tasks; a single coarse space-time attention is not enough.
6. **Things that did not help:** rank-32 weights (bad long forecasts), a fourth level (less accurate), coarse space-time attention (0.006).

## 10. Open questions and possible next steps

- **Combine the strengths of both local variants.** Keep the spectral stages, the channel MLP and the temporal attention at every level. Give a larger share to local processing using WIND-style residual blocks whose 3 x 3 convolution is replaced by the spherical convolution, at levels 0 and 1. This keeps degree-based kernels (good for transfer) and avoids zero padding at the seam and the poles. The expected time is about 1.7 to 2.0 h, not yet measured. The supervisor (Sir Zubair) saw no issue with this direction.
- **Isotropy of the spherical convolution** (raised by the supervisor).
  - Two of the four basis functions are isotropic; the two gradient functions are tied to local north.
  - The operator is therefore equivariant to longitude shifts but not to general rotations. This seems acceptable for weather, which has real preferred directions, and the dense spectral weight is not isotropic either.
  - Weak point: at the pole rows, local north is undefined, so each pole point gets a different frame.
  - Possible tests: an isotropic-only kernel, or adding second-order directional functions.
  - What the DISCO paper says on this has not been checked yet.
- **Single seed.** Differences below about 0.008 are not conclusive. Repeating the best runs with other seeds would need an explicit decision, since the project fixes seed 42.
- **Budget effect.** WIND values at 25,000 steps are interpolated from 45,000-step cosine runs. A cooldown test of the step budget was set up; its results are not in the repository.

## 11. Where things are

- **Paper:** `paper/main.tex` (compiled with `tectonic main.tex`), figures from `paper/make_figures.py`.
- **Models:** `src/spheridiff/models/spheridiff/` (`pooled.py`, `levels.py`, `spectral.py`, `local.py`, `hybrid.py`); WIND in `src/spheridiff/models/wind/`.
- **Sphere code:** `src/spheridiff/sphere/` (`transform.py` least-squares analysis, `slepian.py` kernel Slepian functions).
- **Fine-tuning:** `src/spheridiff/finetune/transfer.py`, `scripts/finetune.py`.
- **Configs:** `configs/model/`, `configs/experiments/` (`pilot_*`, `finetune_*`), `configs/train/pilot_25k.yaml` and `finetune.yaml`.
- **Notebooks:** `notebooks/kaggle_pilot_*.ipynb`, one per experiment stage.
- **Results:** `results/pilot`, `pilot_l31`, `finetune`, `pooled`, `pooled_fixes`, `pooled_noise_coeff`, `local_hybrid`. Checkpoints are local only.
- **Detailed pages:** [models.md](models.md), [training.md](training.md), [final_evaluation.md](final_evaluation.md).
