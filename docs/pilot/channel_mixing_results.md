# Results: Per-Degree Channel Mixing

This page reports the experiment that replaced the mode-mixing spectral weight of the pooled SpherIDiff with per-degree channel mixing. It describes the compared architectures, the shared setup and the results, and it states the limits of the comparison. The model code is described in [models.md](models.md#per-degree-channel-mixing), and the run in [training.md](training.md).

## Contents

1. [Question](#question)
2. [Architectures compared](#architectures-compared)
3. [Shared setup](#shared-setup)
4. [Results](#results)
5. [Reading of the results](#reading-of-the-results)
6. [Limits](#limits)
7. [Source](#source)

## Question

In the earlier pooled models, the spectral stage applied one dense weight W over the spherical harmonic coefficients, shared by all channels. It mixed coefficients with each other but never mixed channels, and it held 73% of the parameters of the pooled model. A rank-32 form of W reached almost the same validation error (0.344 against 0.340), which suggests that most of these parameters were not used well.

The hypothesis tested here is that the same budget does more good as channel mixing that depends on the scale.

## Architectures compared

### Channel mixing (`spheridiff_pooled3_channel`, new)

- **Levels.** Three pooled levels, like a U-Net: grids 32 x 64, 16 x 32 and 8 x 16, with least-squares band limits L = 31, 15 and 7. Pooling keeps the degrees below the band limit of the coarser level, and unpooling synthesises on the finer grid and adds the skip connection.
- **Blocks.** Five blocks: one per level on the way down (levels 0 and 1), one at the bottom (level 2) and one per level on the way up (levels 1 and 0). Base width 64, doubling at each level down.
- **Each block.** A spectral stage, a channel MLP and a temporal attention over the five frames (4 heads), each preceded by an RMS normalisation and added as a residual.
- **Spectral stage.** For coefficients f of degree l and order m, the output is f_lm A(l) + (m/L) f_l,-m B(l). A(l) and B(l) are width x width matrices interpolated with hat functions from 8 knots in the degree (7 at the coarsest level, one per degree). A(l) mixes channels at each degree, and B(l) does the same on the east-west derivative of the field. No learned mode mixing.
- **Other parts.** A spherical convolution at the end of the two blocks of the finest level (four fixed basis functions within about 5.8 degrees), noise-level conditioning after the uplift, and 16-bit spectral stages and pooling.
- **Fine-tuning.** Each level uses the Slepian functions of the patch (L' = 120, 60, 30; M = 175, 44, 11). The pre-trained knots are frozen, new knots cover the unseen degrees and start at zero, and a trainable M x M update dG is added. The pointwise layers, the attention and the spherical convolution are also trained.

### Previous best (`spheridiff_pooled3_disco`)

The same model in every part except the spectral stage. Its spectral stage is a dense K x K weight W shared by all channels (K = 961, 225 and 49 at the three levels), with a per-channel filter kappa_c(l, m) on each coefficient: channel c is mixed with W + diag(kappa_c). On the patch, each channel has its own frozen restricted matrix U_M^T (W + diag(kappa_c)) U_M, and the shared update dG and the pointwise layers are trained.

### WIND-S and WIND-M (baselines)

Our re-implementation of WIND, a U-ViT on the grid. It has four levels: two levels of residual convolution blocks (3 x 3 convolutions with group normalisation) and two levels of Transformer blocks with attention over all points and frames, one skip per level, and 3D rotary position embeddings. Embedding width 64, 4 heads, patch size 1, 32 normalisation groups. WIND-S has widths 64, 128, 128 and 128, and WIND-M 64, 128, 256 and 256. On the patch, all weights are fine-tuned.

### Parameters by part

Measured from the code for 25 channels, 10 auxiliary channels and 5 frames.

| Part | Channel mixing | `pooled3_disco` |
|---|---|---|
| Spectral stages | **1,671,168** (69%) | 2,213,477 (75%) |
| Temporal attention | 426,624 | 426,624 |
| Channel MLP and shortcuts | 276,352 | 276,352 |
| Spherical convolution | 32,896 | 32,896 |
| Noise conditioning | 4,288 | 4,288 |
| Uplift and downlift | 3,929 | 3,929 |
| Normalisations | 640 | 640 |
| **Total** | **2,415,897** | **2,958,206** |

The two models differ only in the spectral stages. In `pooled3_disco` these consist of the dense weights (1,950,693) and the coefficient filters (262,784).

| Part of WIND-S | Parameters |
|---|---|
| Down and up stages (residual and Transformer levels) | 1,218,688 |
| Resampling layers | 1,568,096 |
| Middle Transformer block | 214,592 |
| Input, output and noise embedding | 3,993 |
| **Total** | **3,005,369** |

## Shared setup

| Setting | Value (the same for every model) |
|---|---|
| Global data | ERA5 from WeatherBench 2, 32 x 64 equiangular grid with poles (about 5.6 degrees), 25 channels, train 2010 to 2019 |
| Pre-training | 25,000 steps (`pilot_25k`), batch 16, AdamW at 3e-4, cosine schedule with 5% warmup, gradient clipping 0.8, EMA 0.999, 16-bit mixed precision |
| Validation | 32 windows of 2020, 3 members, 15 DDIM steps |
| Final protocol | `final_v1`: 64 windows of 2021, 5 members, a 20-frame forecast and six reconstruction tasks with MMPS |
| Regional data | 16 x 16 points at 1.5 degrees over South Asia (18.0 to 40.5 N, 58.5 to 81.0 E), train 2017 to 2019 |
| Fine-tuning | 5,000 steps (`finetune`) from the EMA weights, evaluated zero-shot and after fine-tuning |
| Seed and hardware | Seed 42, one T4 GPU on Kaggle per run |

The WIND runs differ from this setup in their length; see [Limits](#limits).

## Results

### Validation during pre-training

Validation nRMSE (lower is better).

| Step | 1,250 | 2,500 | 5,000 | 10,000 | 15,000 | 20,000 | 22,500 | 25,000 |
|---|---|---|---|---|---|---|---|---|
| **Channel mixing** | **0.461** | **0.375** | **0.329** | 0.305 | 0.295 | 0.292 | 0.291 | 0.291 |
| `pooled3_disco` | 0.493 | 0.398 | 0.349 | 0.326 | 0.312 | 0.307 | 0.306 | 0.305 |
| WIND-S (45,000-step run) | 0.747 | 0.538 | 0.401 | 0.321 | 0.285 | 0.268 | 0.264 | 0.260 |
| WIND-M (45,000-step run) | 0.678 | 0.421 | 0.348 | **0.287** | **0.261** | **0.248** | **0.244** | **0.241** |

- Channel mixing is better than `pooled3_disco` at every step, by 0.015 at the end.
- It learns faster than both WIND models at the start. WIND-M passes it at about 7,250 steps and WIND-S at about 12,750 steps, later than for the earlier SpherIDiff models (about 6,500 to 11,000 steps for WIND-S).
- The curve still flattens: it gains only 0.001 between 20,000 and 25,000 steps.
- The WIND models were evaluated every 2,250 steps, so their values are interpolated except at 22,500 steps, where every run has an exact evaluation.

### Size, cost and accuracy of all main models

| Model | Parameters | Training time to 25,000 steps | Seconds per step | Peak memory | Global | Regional | Zero-shot |
|---|---|---|---|---|---|---|---|
| WIND-S | 3,005,369 | **0.78 h** | **0.113** | 1.10 GB | 0.260 | 0.295 | 1.233 |
| WIND-M | 7,336,121 | 0.91 h | 0.131 | | **0.241** | **0.279** | 1.204 |
| SpherIDiff, L = 16 | 1,601,561 | 3.55 h | 0.511 | | 0.312 | 0.358 | **0.515** |
| SpherIDiff, L = 31 (low rank) | **1,589,277** | 5.88 h | 0.849 | | 0.289 | 0.358 | 0.815 |
| Pooled (dense W) | 2,658,238 | 1.24 h | 0.179 | | 0.340 | 0.340 | 0.828 |
| Pooled + filters + noise level | 2,925,310 | 1.12 h | 0.162 | | 0.323 | 0.334 | 0.731 |
| `pooled3_disco` | 2,958,206 | 1.48 h | 0.212 | 1.53 GB | 0.305 | 0.324 | 0.764 |
| Hybrid U-ViT | 2,685,308 | 0.90 h | 0.130 | | 0.370 | 0.329 | 0.965 |
| **Channel mixing** | 2,415,897 | 1.42 h | 0.205 | 1.65 GB | 0.291 | 0.315 | 0.985 |

Global is the validation error after 25,000 steps, regional the validation error on the patch after 5,000 fine-tuning steps, and zero-shot the same regional measure before fine-tuning. The two models at L = 16 and L = 31 were fine-tuned with a regional band limit of 61, the others with 120. The WIND times are those of the 45,000-step runs at step 25,000.

Channel mixing has 18% fewer parameters than `pooled3_disco` and trains 4% faster. It equals the accuracy of the L = 31 model (0.289) in a quarter of its training time. Against WIND-S it has 20% fewer parameters but needs 1.8 times the training time.

### Fine-tuning

| Model | Trainable parameters | Fine-tuning time (5,000 steps) | Seconds per step | Regional after fine-tuning |
|---|---|---|---|---|
| WIND-S | 3,005,369 (all) | 183 s | 0.033 | 0.295 |
| WIND-M | 7,336,121 (all) | 173 s | 0.031 | 0.279 |
| `pooled3_disco` | 809,972 | 190 s | 0.038 | 0.324 |
| **Channel mixing** | 2,481,140 | 268 s | 0.054 | 0.315 |

In channel mixing, 1,671,168 of the trainable parameters are the new knots above the pre-trained band limit and 65,243 are the updates dG. Building the regional model takes about two minutes once, because the Slepian operators of every knot are computed at the start.

### Final protocol on the global test year

Normalised RMSE of the 20-frame forecast and of the unobserved entries of six reconstruction tasks solved with MMPS. Sp.: spatial, temp.: temporal, s-t: spatiotemporal, rec.: reconstruction, down.: downsampling.

| Model | Steps | Forecast | Sp. rec. | Sp. down. | Temp. rec. | Temp. down. | S-t down. | S-t rec. |
|---|---|---|---|---|---|---|---|---|
| **Channel mixing** | 25,000 | 0.499 | 0.484 | 0.396 | 0.267 | 0.280 | 0.437 | 0.377 |
| `pooled3_disco` | 25,000 | 0.485 | 0.506 | 0.407 | 0.286 | 0.284 | 0.446 | 0.398 |
| WIND-S | 45,000 | 0.422 | 0.457 | 0.342 | 0.223 | 0.217 | 0.383 | 0.332 |
| WIND-M | 45,000 | **0.400** | **0.442** | **0.334** | **0.199** | **0.206** | **0.377** | **0.319** |

Channel mixing improves all six tasks over `pooled3_disco`, most clearly spatial reconstruction (0.506 to 0.484) and spatiotemporal reconstruction (0.398 to 0.377). The 20-frame forecast is slightly worse (0.499 against 0.485).

### Final protocol on the regional patch after fine-tuning

| Model | Forecast | Sp. rec. | Sp. down. | Temp. rec. | Temp. down. | S-t down. | S-t rec. |
|---|---|---|---|---|---|---|---|
| **Channel mixing** | 0.438 | 0.339 | 0.251 | 0.264 | 0.248 | 0.285 | 0.287 |
| `pooled3_disco` | 0.458 | 0.364 | 0.260 | 0.268 | 0.243 | 0.296 | 0.308 |
| WIND-S (from 22,500 steps) | 0.407 | 0.340 | 0.236 | 0.250 | 0.226 | 0.280 | 0.276 |
| WIND-M (from 22,500 steps) | **0.391** | **0.328** | **0.229** | **0.237** | **0.214** | **0.272** | **0.266** |

On the patch, channel mixing improves the forecast and five of the six tasks over `pooled3_disco`. It equals WIND-S on spatial reconstruction (0.339 against 0.340) and comes close on spatiotemporal downsampling (0.285 against 0.280).

### Zero-shot transfer

| Model | Regional validation, zero-shot | Zero-shot forecast (final protocol) |
|---|---|---|
| WIND-S | 1.233 | 1.356 |
| `pooled3_disco` | **0.764** | **0.910** |
| **Channel mixing** | 0.985 | 1.013 |

Before fine-tuning, channel mixing transfers worse than `pooled3_disco`, but still better than WIND-S. The cause was not tested. Both SpherIDiff models have no spectral response above the pre-trained band limit before fine-tuning, so that alone does not explain the difference.

## Reading of the results

1. **The hypothesis is supported.** Using the spectral parameters for scale-dependent channel mixing instead of mode mixing gave a better model with fewer parameters, globally and on the patch.
2. **The gap to WIND-S became smaller.** Globally it fell from 0.045 to 0.031 (0.027 at the exact 22,500-step point), and regionally from 0.029 to 0.020.
3. **The gains are mostly in reconstruction.** The forecast did not improve globally, while every reconstruction task did.
4. **Zero-shot transfer became weaker,** although it remains clearly better than that of WIND.
5. **The cost is close to that of the previous best,** and still about 1.8 times the training time of WIND-S.

## Limits

- **WIND reference.** No WIND model trained for 25,000 steps existed. The WIND validation values come from the middle of 45,000-step cosine schedules, their global tables use the 45,000-step weights, and their fine-tuning started from 22,500 steps. The exact gaps to WIND are therefore uncertain. New runs of WIND-S and WIND-M under the same settings as SpherIDiff are prepared in `notebooks/kaggle_pilot_wind_25k.ipynb`.
- **One seed.** Two runs of the same model differed by about 0.008 at 1,000 steps. The global gain over `pooled3_disco` (0.015) is about twice this and is supported by all six tasks. The regional gain (0.009) is at the edge of it.
- **One resolution and one region,** as in the rest of the pilot.

## Source

The results of channel mixing come from `channel_mixing_results.zip`, produced by `notebooks/kaggle_pilot_channel_mixing.ipynb`. The reference values come from `results/local_hybrid` (`pooled3_disco` and the hybrid), `results/pooled`, `results/pooled_noise_coeff`, `results/pilot_l31`, `results/pilot` and `results/finetune`. A local test of 1,000 steps on the RTX 4050 gave the same ordering: 0.469 against 0.535 for `pooled3_disco`.
