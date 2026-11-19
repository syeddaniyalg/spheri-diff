# Smoke Test Results

Results of one smoke test run on Kaggle (two GPUs, one model per GPU). The stages and thresholds are described in [docs/pilot/smoke.md](../../docs/pilot/smoke.md). All stages passed for both models.

## Summary

| | smoke_wind_s | smoke_spheridiff_b4 |
|---|---|---|
| parameters | 3,005,369 | 1,601,561 |
| overfit loss ratio | 0.047 | 0.028 |
| short run loss ratio (2,000 steps) | 0.126 | 0.070 |
| normalised RMSE at step 2,000 | 0.619 | 0.411 |
| normalised CRPS at step 2,000 | 0.285 | 0.232 |
| channel-mean reference, normalised RMSE | 0.987 | 0.987 |
| persistence reference, normalised RMSE | 0.396 | 0.396 |
| resume gap ratio | 1.2e-8 | 0.0 |
| samples per second | 148.8 | 19.7 |
| median seconds per step | 0.108 | 0.810 |
| peak memory (GB) | 1.19 | 9.38 |
| training time of the 2,000 steps | 216 s | 1,618 s |
| projected pilot total (45,000 steps, with margin) | 1.6 h | 12.4 h |

Both models are far below the channel-mean reference. Both are still above the persistence reference on the normalised RMSE at 2,000 steps, which is expected this early, and SpherIDiff is much closer to it. SpherIDiff learns more per step and per sample, but each step costs about 7.5 times more, so WIND is ahead per second of training. The projected SpherIDiff pilot time is longer than a 12-hour session, which is why the pilot notebook stops on a time limit and resumes.

## Contents

- `status.json`: job status, device and duration of each model run.
- `evaluation_curves.png`: validation loss, normalised RMSE and normalised CRPS against steps, samples seen and training seconds.
- `training_loss.png`: the training loss of both models.
- `<model>/smoke_report.json`: the result of every stage and the timing estimate.
- `<model>/summary.json`: steps, time, final loss and peak memory of the short run.
- `<model>/evaluations.jsonl` and `steps.jsonl`: the evaluation records and the per-step training log.
- `<model>/config_snapshot.json`: the configuration of the run.
- `<model>/checkpoint.pt`, `final.pt` and `ema_weights.pt`: the training checkpoint, the final weights and the EMA weights. These files are large (about 280 MB in total) and are not tracked by git.
- `<model>_overfit/` and `<model>_resume_a/`, `<model>_resume_b/`: the runs of the overfit and resume stages, with the same files.
- `_logs/`: the console logs of each model and of the parallel runner.
