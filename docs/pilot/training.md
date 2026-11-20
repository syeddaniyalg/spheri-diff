# Training and Evaluation

This page describes how the pilot models are trained and evaluated. All settings are in `configs/`, and both models use the same data, loss, schedule, batch stream and evaluation.

## Training

| Setting | Value |
|---|---|
| Objective | weighted x0 loss with an independent noise level per frame |
| Loss weights | area weights, channel weights (pressure level over the sum of the 13 source levels, 1.0 for 2 m temperature, 0.1 for other surface variables), and (std / diff_std)^2 |
| Optimiser | AdamW, weight decay 0 |
| Learning rate | 3e-4 with 5% warmup and cosine decay to 1e-6 (a provisional choice, the published model used 1e-4) |
| Gradient clipping | 0.8 |
| EMA | decay 0.999 with a warm-up, used for every evaluation |
| Batch size | 16 |
| Steps | 45,000 (provisional, to be fixed from the smoke test timing) |
| Precision | fp16 with a gradient scaler |

The noise level of every step is drawn from a generator seeded by the step number, and the batches come from a seeded stream that can start at any step. A run that stops and resumes therefore follows the same path as an uninterrupted run.

Every experiment uses the same fixed seed, 42 (`DEFAULT_SEED` in `src/spheridiff/common/seed.py`), for the model initialisation, the batch stream, the training noise, the evaluation windows and noise, the sensor mask of the sparse task and the bootstrap. A new experiment inherits it by default, and a test checks that no configuration sets another value.

Every step is logged to `steps.jsonl` (loss, learning rate, gradient norm, step and data time, samples per second, memory), and the progress bar shows the live loss, learning rate, samples per second and gradient norm.

## Saved files

Each run writes to its own folder (`artifacts/runs/<run name>/`, or `/kaggle/working/runs/<run name>/` with `--runs-root`).

| File | When | Contents |
|---|---|---|
| `config_snapshot.json` | at the start | experiment, train settings, model spec, parameter count, data splits, library version, GPU |
| `steps.jsonl` | every step | the step metrics above |
| `evaluations.jsonl` | step 0 and every evaluation interval | validation loss and forecast scores |
| `checkpoint.pt` | every `checkpoint_every` steps, replaced each time | everything needed to resume |
| `ema_weights.pt` | with each checkpoint | the EMA weights used for evaluation |
| `final.pt` | once, when the last step is done | the final training state |
| `milestone_<step>.pt` | at 25%, 50% and 75% of the steps (`milestones` in the train config), never replaced | the full training state, so a short learning-rate decay can be started from that point |
| `windows.jsonl` | with every evaluation | per-window normalised squared error and CRPS of the forecast evaluation, with the window indices, for paired comparisons |
| `summary.json` | at the end | steps, time, speed, final loss, memory |

A run resumes from `checkpoint.pt` automatically. To collect everything for download, `--package results.zip` on `scripts/train.py` (or `scripts/package_runs.py`) packs all run folders into one archive.

```
python scripts/train.py --experiment pilot_wind_s
python scripts/train.py --experiment pilot_spheridiff_b4
```

## Comparing learning curves

Both models use the same schedule and total steps, so they can be compared at the same step, and the cosine schedule makes a mid-run value a statement about the model at that point of training, not about a shorter run. `scripts/compare_runs.py --run <run> --references <run> ...` (also run by the Kaggle notebooks) reports for a run against each reference:

- the reference final value as the target, and the first step, sample count and training time at which the run reaches it and stays at or below it for two evaluations in a row;
- the values at 5%, 10%, 25%, 50% and 100% of training;
- the mean over the whole curve (trapezoid rule over the steps);
- the paired difference of the window-level normalised RMSE at the last common step, with a 95% bootstrap interval over the evaluation windows (2,000 resamples).

The results are written to `comparison_<run>_vs_<reference>.json` and `comparison_summary.txt`. The comparison holds at the shared learning rate, and a mid-run value is not the result of a short fully decayed run. That would need a run or a decay branch from a milestone checkpoint.

## Time-matched WIND-L run

The main pilot compares the models at equal steps and samples, which favours the model that is slower per sample. To test whether a lead of SpherIDiff remains when WIND gets the same wall-clock time, the largest WIND model is trained once more as `pilot_wind_l_long` (`configs/experiments/pilot_wind_l_long.yaml`, train config `pilot_long`).

- **Setting.** It differs from the pilot only in length: 135,000 steps instead of 45,000, with evaluations and checkpoints every 6,750 steps, and its own full cosine schedule. The data, batch size, learning rate, EMA, evaluation, seed and milestone fractions are the same, and a test checks this.
- **Time match.** The step count is an estimate: about 4.6 hours at an assumed 130 samples per second for `wind_l` (not measured on the T4), against about 5.3 hours projected for SpherIDiff. It should be adjusted when the pilot timing is known.
- **Separate storage.** It is run by its own notebook, `notebooks/kaggle_pilot_timematched.ipynb`, in its own Kaggle session, with its own runs folder and its own archive, `pilot_timematched_results.zip`. Its results are not mixed with the files of the main pilot.
- **Comparison.** When the output of the main pilot is attached as a Kaggle input, the notebook compares the run with `pilot_spheridiff_b4` and `pilot_wind_l`. Here the paired difference uses the last evaluation of each run, because the step counts differ, and the figures show the references together with the run.
- **Smoke test.** It is not part of the smoke test.

## Shorter SpherIDiff run

To test whether SpherIDiff keeps up with WIND-L when both get about the same wall-clock time, `spheridiff_b4` is also trained for fewer steps as `pilot_spheridiff_b4_short` (`configs/experiments/pilot_spheridiff_b4_short.yaml`, train config `pilot_short`).

- **Setting.** It differs from the pilot only in length: 11,625 steps instead of 45,000, with evaluations and checkpoints every 375 steps, and its own full cosine schedule.
- **Time match.** In the main pilot, `wind_l` trained for 6,057 s and SpherIDiff needed 0.511 s per step, which gives about 11,850 steps. The value is rounded down to a multiple of 375.
- **Band limit.** It uses the band limit of the main pilot (L = 16), so that the comparison stays controlled.
- **Notebook.** It is run by `notebooks/kaggle_pilot_variants.ipynb`, together with the smaller variants.

## Smaller SpherIDiff variants

The variants `spheridiff_xs`, `spheridiff_s` and `spheridiff_m` (0.27M, 0.35M and 0.60M parameters, see [models.md](models.md)) are trained for the same 45,000 steps as the main pilot. They are run by `notebooks/kaggle_pilot_variants.ipynb` in a separate Kaggle session, with its own runs folder and its own archive, `pilot_variants_results.zip`.

- **Placement.** The three variants are three parallel processes on the first GPU, because their memory use is far below the memory of the card. The shorter run of `spheridiff_b4` uses the second GPU. The runner places each queue on a GPU that is named in the notebook, so several queues can share one GPU.
- **Throughput.** Whether three processes on one GPU finish faster than one after another has not been measured. The processes share the GPU in turns.
- **Comparison.** When the output of the main pilot is attached as a Kaggle input, the notebook compares every run with the WIND runs and the main SpherIDiff run. The paired difference uses the last evaluation of each run, because the step counts differ.

## SpherIDiff at band limit 31

The four SpherIDiff sizes are trained again with band limit L = 31 and the amendments described in [models.md](models.md#spheridiff-at-band-limit-31). They are run by `notebooks/kaggle_pilot_l31.ipynb` in a separate Kaggle session, with its own runs folder and its own archive, `pilot_l31_results.zip`.

- **Budget.** Every run is trained for 25,000 steps (`configs/train/pilot_25k.yaml`), the fixed comparison budget. Evaluation and checkpoints come every 1,250 steps, so the curve has 20 points as in the main pilot. All other settings, including the seed 42, are those of the pilot.
- **Placement.** Each GPU trains one run at a time. The first GPU trains `xs` and then `b4`, and the second GPU trains `s` and then `m`, which pairs the smallest run with the largest. The two queues run in parallel, and the smaller run of each pair comes first. With one GPU, both queues share it.
- **Final evaluation.** When all training jobs have ended, every run is evaluated under the fixed protocol `final_v1` on the same two queues.
- **Comparison.** When the output of the main pilot is attached as a Kaggle input, every run is compared with the WIND runs and with the main SpherIDiff run at L = 16. These references were trained for 45,000 steps, so the paired difference of the final errors favours them. Their errors at 25,000 steps appear among the errors at fixed fractions of training and in the evaluation curves.

## Cooldown test of the step budget

Under the cosine schedule the late part of a learning curve flattens partly because the learning rate decays, so the curve alone does not show how many steps a run needed. The cooldown test measures this from the milestone checkpoints of the main pilot, without training the runs again.

- **Branches.** From each milestone (11,250, 22,500 and 33,750 steps), training continues for 15% of the milestone step. The learning rate starts at its value on the cosine schedule at the milestone and falls to the minimum learning rate with the (1 - sqrt) shape (`cooldown_rate_at` in `src/spheridiff/training/schedule.py`). Each branch is a run folder of its own, for example `pilot_spheridiff_b4_cooldown_022500`, and is evaluated once, at its last step, with the evaluation used during the pilot.
- **Same stream.** A branch restores the model, optimizer, scaler and EMA of the milestone, keeps the seed 42 and continues the batch stream and the training noise from the milestone step. It differs from the full run only in the learning rate schedule, and a test checks that the first branch step has the same loss as the full run at that step.
- **Reading.** Each branch is compared with the full 45,000-step run by the paired difference of the normalised RMSE over the 32 validation windows, with a 95% bootstrap interval. A branch matches the full run when the upper end of the interval is at most 0.002. The smallest matching branch end is the budget that the run needed. The table is written to `cooldown_summary.txt` and `.json`, and the figure to `cooldown_curves.png`.
- **Script.** `scripts/cooldown.py` runs the branches of one experiment in one process, so the data is loaded once. A branch whose summary reaches its last step is skipped, and `--keep-only-weights` deletes the large checkpoints of a finished branch.
- **Notebook.** The test is run by `notebooks/colab_pilot_cooldown.ipynb` on Google Colab with one T4 GPU. It reads one archive from Google Drive with the ERA5 cache and, for each pilot run, the milestones, the configuration snapshot and the evaluation logs, and it writes its results archive back to Drive. At the speeds measured in the pilot, the twelve branches take about two and a half hours.

## Running on Kaggle

The notebooks `notebooks/kaggle_pilot.ipynb` and `notebooks/kaggle_smoke.ipynb` run the jobs in the background, as a saved version that Kaggle executes from start to end. Both use the runner in `src/spheridiff/kaggle/`.

- **Parallel GPUs.** The three WIND sizes form one queue for the first GPU, and SpherIDiff forms a second queue for the second GPU. Each job is a separate process with `CUDA_VISIBLE_DEVICES` set to its GPU. With one GPU, the queues are joined and run one after another.
- **Logs.** Every job writes its own log to `runs/_logs/<job>.log`. The notebook prints every line of every running job as soon as it appears, with the elapsed time, GPU and job name, and appends the same lines to `runs/_logs/parallel.log`. The progress bars refresh at most once per second (`TQDM_MININTERVAL=1`), so a long run produces a large log.
- **Time limit.** The pilot notebooks set no time limit, and the jobs run until Kaggle ends the session. The runner accepts an optional deadline: when one is given, as in the smoke notebook (`SESSION_HOURS`), jobs receive `--max-seconds`, so they stop with a resumable checkpoint about ten minutes before the deadline, and the runner ends any process that is still running at the deadline.
- **Results.** The runs, the status file, the figures and the logs are written under `/kaggle/working/runs`, and one archive (`pilot_results.zip` or `smoke_results.zip`) is written next to it. The run and the packing share one cell with `try` and `finally`, so the archive is produced when a job fails or the cell is interrupted. The archive is also refreshed every 30 minutes.

Some Kaggle facts could not be confirmed from the official documentation. Secondary sources give a session limit of 9 or 12 hours, a weekly GPU quota of about 30 hours, and an output folder limit of about 20 GB. They also say that a saved version is a batch session without the inactivity limit, and that its files in `/kaggle/working` are kept when it ends. It is not confirmed what is kept when Kaggle stops a version at its time limit, which is why the archive is refreshed during the run and the deadline is set below the limit. The values should be checked in the Kaggle documentation.

## Evaluation during training

At step 0 and at every 5% of the steps, the EMA model is evaluated on fixed validation windows with fixed seeds. The log `evaluations.jsonl` holds:

- the validation loss at fixed noise levels;
- the forecast RMSE and CRPS for six fields at the leads of 6, 12, 18 and 24 hours;
- the normalised RMSE and CRPS averaged over all channels (the headline curves);
- at step 0, two reference scores: persistence and the channel mean.

Curves of several runs are drawn with `scripts/plot_runs.py`.

## Full evaluation after training

The fixed protocol used for the final comparison of the models is described in [final_evaluation.md](final_evaluation.md). The lighter evaluation below (`scripts/evaluate.py`, with the settings named in the experiment file) is kept for quick checks, and the Kaggle notebooks use the fixed protocol.

`scripts/evaluate.py` loads the EMA weights and runs, on the test year:

- forecasting with rolled windows up to 5 days, reporting RMSE, CRPS and spread-skill ratio;
- spatial downscaling by a factor of 4, temporal downscaling from a one-frame mean, and sparse reconstruction from 5% of the grid points, each solved with moment matching posterior sampling (15 DDIM steps, 2 iterations of the linear solver (GMRES), observation variance 0.0015);
- for each task, the RMSE of the ensemble mean and of a simple reference (upsampling, repetition, or zeros away from the sensors).

```
python scripts/evaluate.py --experiment pilot_wind_s
```
