# Pilot Results at Band Limit 31

Results of the four SpherIDiff sizes trained with band limit L = 31 for 25,000 steps on Kaggle (two GPUs: `xs` then `b4` on the first, `s` then `m` on the second), followed by the final evaluation under `final_v1`. The setup is described in [docs/pilot/models.md](../../docs/pilot/models.md#spheridiff-at-band-limit-31) and [docs/pilot/training.md](../../docs/pilot/training.md#spheridiff-at-band-limit-31).

## Summary at 25,000 steps

| Run | Parameters | Training time | Validation nRMSE | Validation CRPS |
|---|---|---|---|---|
| `pilot_spheridiff_l31_xs` | 264,987 | 1.62 h | 0.3662 | 0.1854 |
| `pilot_spheridiff_l31_s` | 335,901 | 2.25 h | 0.3668 | 0.1865 |
| `pilot_spheridiff_l31_m` | 587,805 | 3.52 h | 0.3192 | 0.1612 |
| `pilot_spheridiff_l31_b4` | 1,589,277 | 5.88 h | 0.2891 | 0.1436 |

At the same step, the main pilot runs had a validation nRMSE of 0.3121 (SpherIDiff b4 at L = 16) and 0.2598, 0.2408 and 0.2437 (WIND-S, M and L). The final evaluation table in `final_evaluation_summary.txt` compares the runs with the main pilot runs, which were trained for 45,000 steps.

## Contents

The files follow the layout of [the main pilot results](../pilot/README.md#contents). The checkpoints (`*.pt`) are kept locally and are not tracked by git.
