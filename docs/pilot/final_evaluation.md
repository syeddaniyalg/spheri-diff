# Final Evaluation

This page describes the final evaluation of the pilot models. It runs after the training, on a held-out test set, under one fixed protocol, so that models trained later can be evaluated under exactly the same conditions and compared directly.

## Held-out test set and fixed protocol

- **Test set.** The test split is the year 2021. No training step, validation, checkpoint choice or setting was based on it. The evaluated weights are the EMA weights at the end of training.
- **Protocol file.** All settings are in `configs/protocol/final_v1.yaml`: the split, the number of windows, the number of ensemble members, the sampler settings, the seed, the six reported fields, the guidance settings and the definition of every task. A SHA-256 digest of the file content is stored in every result.
- **Fixed windows.** The windows are evenly spaced over the test year and are the same for every model. Each result stores the window indices and their start times.
- **Controlled comparison.** The summary table refuses to combine results with different digests or different windows. A new variant is evaluated by running the same command with the same protocol name. To change the protocol, a new file (for example `final_v2.yaml`) is added, and old results are not mixed with it.
- **Precision and seed.** The evaluation runs in fp32, with the standard seed 42 for every window and member.

```
python scripts/final_evaluation.py --experiments pilot_wind_s pilot_spheridiff_b4 --protocol final_v1
python scripts/final_summary.py --runs pilot_wind_s pilot_spheridiff_b4
```

## What is evaluated

**Forecasting.** For each of the 64 windows, 5 ensemble members are generated over a 5-day lead (21 frames, rolled in windows of four new frames) with 15 DDIM steps. The result holds the RMSE, CRPS and spread-skill ratio for six fields at each lead, the normalised RMSE and CRPS averaged over all channels, and the normalised error of every window.

**Reconstruction tasks.** Each task is defined by an observation operator applied to the true five-frame window, and the model has to reconstruct the window from the observation.

| Task | Observation |
|---|---|
| spatial reconstruction | 5% of the grid points (sensors), at all frames and channels |
| spatial downsampling | the field averaged over blocks of 4 x 4 points |
| temporal reconstruction | the first and last frames only |
| temporal downsampling | the mean over the five frames |
| spatiotemporal downsampling | the mean over the five frames of the field averaged over blocks of 4 x 4 points |
| spatiotemporal reconstruction | 5% of all space-time points, shared by all channels |

**Methods.** Every task is solved in three ways, with the same sampler settings (15 DDIM steps, 5 members, all frames starting from noise):

- **MMPS:** moment matching posterior sampling as in the WIND code (azula): the denoiser output is corrected by the Tweedie covariance term, with the system solved by 2 iterations of a Krylov solver (GMRES, the default of the released code; the paper text names conjugate gradient, so `solver` in the protocol can be set to `cg`) and an observation variance of 0.0015. For SpherIDiff the noise covariance Q_s = (4 pi / K) S^T S of the spectral noise enters the system matrix and the correction, as derived in the proposal. For Gaussian noise Q_s is the identity, which gives the WIND form.
- **Native denoising (`native`):** the same DDIM loop without the likelihood term, as in the proposal. The observation does not enter, so this is the reference for how much the guidance contributes.
- **Native denoising from the observation (`native_init`):** the same loop without the likelihood term, started from a baseline estimate built from the observation (nearest-neighbour upsampling, linear interpolation between the observed frames, repetition of the mean, or the observed values with the channel mean elsewhere). The estimate is noised to level 0.5 and denoised in 8 steps. The level is set in the protocol file.

Each task also has a reference without any model, which uses the same baseline estimate.

## Metrics

For every task and method, the result stores:

- `nrmse`: the normalised RMSE of the ensemble mean over all entries;
- `nrmse_unobserved`: the same, over the unobserved entries only. For the three mask tasks this is the main number, because the observed entries are given. For the two averaging tasks it equals `nrmse`;
- `consistency`: the RMSE between the observation operator applied to the result and the observation itself;
- `ncrps`: the normalised CRPS of the ensemble (not defined for the reference);
- `rmse`: the RMSE of each reported field in physical units;
- `window_nmse`: the normalised squared error of each window, for paired comparisons.

The summary table lists, for each run, the number of parameters, the trained steps, whether training finished, the forecast error and CRPS, and the `nrmse_unobserved` of every task and method, with the reference as the last row.

## Cost

Measured locally for 64 windows before the third method and the sixth task were added; the real cost is now higher, and the guided runs dominate it (the T4 may differ): about 8 minutes for the smallest WIND size and 11 minutes for the largest at batch size 8, and about 81 minutes for SpherIDiff at batch size 4. The protocol uses batch size 4, which may make the WIND runs slower than these figures. The Kaggle notebooks start the evaluation after the training, and write `final_evaluation.json` into each run folder and the table into the output.
