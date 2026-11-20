# Pilot Results

Results of the pilot run on Kaggle (two GPUs). The three WIND sizes were trained one after another on the first GPU, and SpherIDiff (`spheridiff_b4`) was trained on the second GPU. Every model was trained for 45,000 steps and then evaluated under the fixed final protocol `final_v1` on the test year. The setup is described in [docs/pilot/README.md](../../docs/pilot/README.md), and the final evaluation in [docs/pilot/final_evaluation.md](../../docs/pilot/final_evaluation.md).

## Summary

| | pilot_wind_s | pilot_wind_m | pilot_wind_l | pilot_spheridiff_b4 |
|---|---|---|---|---|
| parameters | 3,005,369 | 7,336,121 | 12,238,201 | 1,601,561 |
| training time | 1.4 h | 1.6 h | 1.7 h | 6.4 h |
| final forecast normalised RMSE | 0.4217 | 0.3999 | 0.4029 | 0.4551 |
| final forecast normalised CRPS | 0.2127 | 0.1999 | 0.2025 | 0.2426 |
| spatial reconstruction, MMPS | 0.457 | 0.442 | 0.439 | 0.506 |
| spatial downsampling, MMPS | 0.342 | 0.334 | 0.334 | 0.414 |
| temporal reconstruction, MMPS | 0.224 | 0.199 | 0.204 | 0.344 |
| temporal downsampling, MMPS | 0.217 | 0.206 | 0.207 | 0.363 |
| spatiotemporal downsampling, MMPS | 0.383 | 0.377 | 0.378 | 0.436 |
| spatiotemporal reconstruction, MMPS | 0.332 | 0.319 | 0.321 | 0.433 |

The task rows are the normalised RMSE on the unobserved entries. The full table with all three methods (MMPS, native and native from the noised baseline) and the task reference row is in `final_evaluation_summary.txt`. In the training comparison, SpherIDiff ends at a normalised RMSE of 0.307 on the validation windows, against about 0.23 to 0.25 for the WIND sizes, and it does not reach their final error. The comparison is in `comparison_summary.txt`.

## Contents

- `status.json`: job status, GPU and duration of every training and evaluation job.
- `final_evaluation_summary.txt` and `.json`: the final evaluation table of all runs.
- `comparison_summary.txt` and `comparison_*.json`: the comparison of SpherIDiff with each WIND size during training.
- `evaluation_curves.png`: validation normalised RMSE and CRPS against steps, samples seen and training seconds.
- `training_loss.png`: the training loss of all runs.
- `<run>/summary.json`: steps, training time, speed, final loss and peak memory.
- `<run>/final_evaluation.json`: the result of the final protocol for the run.
- `<run>/evaluations.jsonl`, `windows.jsonl` and `steps.jsonl`: the evaluation records, the per-window errors and the per-step training log.
- `<run>/config_snapshot.json`: the configuration of the run.
- `_logs/`: the console log of every job and the log of the parallel runner (`parallel.log` is about 39 MB).

- `<run>/checkpoint.pt`, `final.pt`, `ema_weights.pt` and `milestone_<step>.pt`: the training checkpoint, the final weights, the EMA weights and the checkpoints at steps 11,250, 22,500 and 33,750. These files are large (about 1.9 GB in total) and are not tracked by git.
