# Smoke Test

The smoke test checks, for each pilot model, that the implementation is complete, that it learns, and how long the pilot run will take. It is run with `scripts/smoke.py` or with the Kaggle notebook `notebooks/kaggle_smoke.ipynb`.

```
python scripts/smoke.py --experiment smoke_wind_s
python scripts/smoke.py --experiment smoke_spheridiff_b4
```

## Stages

| Stage | What it does | Pass condition |
|---|---|---|
| `static` | builds the model and calls it once | the output has the input shape and starts at exactly zero |
| `overfit` | 500 steps on 8 fixed windows | the final loss is below 60% of the initial loss and the error is below the channel-mean reference |
| `short` | 2,000 steps with evaluations every 500 steps | the loss falls by at least 10%, fewer than 5% of the steps are skipped, and the error is below the channel-mean reference |
| `resume` | 6 steps in one run against 3 plus 3 after a resume | the final weights agree within 10% of the distance the weights moved |
| `tasks` | runs forecasting and the three guided tasks on two windows | every path runs and every value is finite |
| `timing` | extrapolates the measured speed to the pilot settings | reports samples per second, hours of training and evaluation, and the total with a 20% margin |

The thresholds are in `src/spheridiff/smoke/criteria.py`. They are first estimates, because no smoke run has been made yet.

## Outputs

Each model writes its logs, checkpoints and `smoke_report.json` to its run folder. The notebook runs both models, on separate GPUs when two are available, plots the logged curves of both, and packs `/kaggle/working/runs` with the console logs and figures into `smoke_results.zip` for download.

## Local GPU check

A short test on the local GPU runs a few fp16 steps, one periodic evaluation and every evaluation path with the real data. It is not part of the default test run.

```
python -m pytest -m gpu
```
