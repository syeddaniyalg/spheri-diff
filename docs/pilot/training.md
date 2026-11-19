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

## Running on Kaggle

The notebooks `notebooks/kaggle_pilot.ipynb` and `notebooks/kaggle_smoke.ipynb` run the jobs in the background, as a saved version that Kaggle executes from start to end. Both use the runner in `src/spheridiff/kaggle/`.

- **Parallel GPUs.** The three WIND sizes form one queue for the first GPU, and SpherIDiff forms a second queue for the second GPU. Each job is a separate process with `CUDA_VISIBLE_DEVICES` set to its GPU. With one GPU, the queues are joined and run one after another.
- **Logs.** Every job writes its own log to `runs/_logs/<job>.log`. The notebook prints every line of every running job as soon as it appears, with the elapsed time, GPU and job name, and appends the same lines to `runs/_logs/parallel.log`. The progress bars refresh at most once per second (`TQDM_MININTERVAL=1`), so a long run produces a large log.
- **Time limit.** `SESSION_HOURS` sets the session length (9 hours by default, see below). Jobs receive `--max-seconds`, so they stop with a resumable checkpoint about ten minutes before the deadline, and the runner ends any process that is still running at the deadline.
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
- spatial downscaling by a factor of 4, temporal downscaling from a one-frame mean, and sparse reconstruction from 5% of the grid points, each solved with moment matching posterior sampling (15 DDIM steps, 2 conjugate gradient iterations, observation variance 0.0015);
- for each task, the RMSE of the ensemble mean and of a simple reference (upsampling, repetition, or zeros away from the sensors).

```
python scripts/evaluate.py --experiment pilot_wind_s
```
