# SpherIDiff

SpherIDiff (Spherical Weather Inverse Diffusion Model) is a diffusion architecture with strong spherical inductive biases for atmospheric data. Its main operations act on spherical harmonic coefficients, which are defined on the sphere itself, and a Slepian projection reduces the number of coefficients the model processes.

The model is trained once on a global grid. It is then applied without task-specific retraining to forecasting, spatial and temporal downscaling, reconstruction from sparse observations, and spatiotemporal super-resolution, using Moment Matching Posterior Sampling (MMPS) at inference. For a regional domain at a different resolution, it is fine-tuned by adapting only a small part of its spectral weights. The project studies how well this transfers across domains and resolutions, and how many parameters and how much computation it needs.

## Index

### Pilot Experiment

A small first experiment with SpherIDiff, run before a full experiment. It includes a comparison with the WIND baseline. See [docs/pilot/README.md](docs/pilot/README.md).

- **Dataset**: the ERA5 data at 5.6 degrees used by the pilot, with its contents, splits, size and the commands to build it. See [docs/pilot/README.md](docs/pilot/README.md#dataset).
- **Models**: the model sizes of the pilot and the design choices behind them. See [docs/pilot/models.md](docs/pilot/models.md).
- **Training and evaluation**: the training settings, the saved files, the parallel background runs on Kaggle, the separate time-matched WIND-L run, the evaluation during training, and the full evaluation after training. See [docs/pilot/training.md](docs/pilot/training.md).
- **Final evaluation**: the fixed protocol on the held-out test year, for forecasting and five reconstruction tasks. See [docs/pilot/final_evaluation.md](docs/pilot/final_evaluation.md).
- **Smoke test**: the short checks that precede the pilot run, with a Kaggle notebook. See [docs/pilot/smoke.md](docs/pilot/smoke.md).

### Results

Small result files of the experiments. See [results/README.md](results/README.md).
