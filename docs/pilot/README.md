# Pilot Experiment

The pilot experiment is a small first experiment with SpherIDiff, run before a full experiment. It also trains the WIND model as a baseline on the same data, so that the two can be compared.

## Index

- **Dataset**: the data used by the pilot, summarised in [Dataset](#dataset) below. The source, channels, cache layout and script options are in [docs/data/era5_5p6deg.md](../data/era5_5p6deg.md).
- **Models**: the three baseline sizes and the SpherIDiff configuration, with their parameter counts and the design choices that were made. See [models.md](models.md).
- **Training and evaluation**: the shared training settings, the saved files, the parallel background runs on Kaggle, the separate time-matched WIND-L run, the evaluation during training, and the full evaluation after training. See [training.md](training.md).
- **Final evaluation**: the fixed protocol on the held-out test year, covering forecasting and five reconstruction tasks, each with and without guidance. See [final_evaluation.md](final_evaluation.md).
- **Smoke test**: the short checks of both models before the pilot run, and the Kaggle notebook that runs them. See [smoke.md](smoke.md).
- **Comparison with WIND**: the baseline comparison, summarised in [Comparison with WIND](#comparison-with-wind) below.

## Dataset

The pilot uses ERA5 reanalysis from WeatherBench 2 on a coarse grid of 32 latitudes by 64 longitudes (about 5.6 degrees). The data is public and needs no account.

| Item | Value |
|---|---|
| Grid | 32 x 64, equiangular, with poles |
| Time step | 6 hours |
| Years | 2010 to 2021 |
| Splits | train 2010 to 2019, validation 2020, test 2021 |
| Window | 5 consecutive frames |
| Predicted channels | 25 |
| Auxiliary channels | 10 (not predicted) |
| Size of the local copy | about 3.3 GB |

Precipitation is transformed with log10(1000 x + 1). All channels are normalised with statistics from the training years only.

The local copy is stored in `artifacts/data/era5_5p6deg/`, which is not tracked by git. The steps to build it, and the options of the build script, are in [docs/data/era5_5p6deg.md](../data/era5_5p6deg.md#how-to-download).

## Comparison with WIND

WIND is a published weather diffusion model used here only as a baseline. In the pilot, both models are trained on this dataset and see the same seeded stream of batches, so that differences in results come from the models and not from the data. WIND is trained at three sizes, which gives a scaling reference for placing the SpherIDiff result.
