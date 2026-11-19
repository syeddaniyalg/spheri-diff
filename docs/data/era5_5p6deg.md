# ERA5 at 5.6 Degrees

This document describes the dataset defined by `configs/data/era5_5p6deg.yaml`, where it comes from, and how to build the local copy.

## Source

The data is ERA5 reanalysis from WeatherBench 2, already regridded to a coarse equiangular grid with poles.

- Store: `gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr`
- Public address: `https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr`
- Grid: 32 latitudes from -90 to 90 degrees (spacing 5.8065 degrees) and 64 longitudes (spacing 5.625 degrees).
- Time: 6-hourly, from 1959-01-02 00:00 to 2021-12-31 18:00.
- Levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa.

The store is public and needs no account. Its arrays are stored with all 13 levels in each chunk, so reading a few levels still reads all of them.

## Selection

- Channels (25): geopotential, temperature, specific humidity, and the u and v wind components at 250, 500, 700 and 850 hPa (20 channels), plus 2 m temperature, mean sea level pressure, 10 m u and v wind, and 6-hour precipitation (5 channels).
- Static fields: land-sea mask, soil type, and surface geopotential.
- Splits: train 2010 to 2019, validation 2020, test 2021.
- Window: 5 consecutive frames, 6 hours apart.

All choices are set in `configs/data/era5_5p6deg.yaml`.

## Local copy

The local copy is built by `scripts/prepare_data.py` and stored in `artifacts/data/era5_5p6deg/`. This folder is not tracked by git. It is empty until the download is run.

```
artifacts/data/era5_5p6deg/
    frames/<year>.npy    raw frames, shape (steps, 25, 32, 64), float32
    times/<year>.npy     seconds since 1970-01-01, one value per step
    static.npz           latitudes, longitudes, and the three static fields
    stats.npz            channel mean, std, and 6-hour difference std (training years only)
```

Frames are stored without any transform. The precipitation transform log10(1000 x + 1), applied to the channels listed in the config, and the normalisation are applied when the data is loaded, so the statistics can change without a new download.

Expected sizes:

- Data read from the source: about 9.4 GB for the 12 years.
- Finished local copy: about 3.3 GB.

## How to download

Run these steps on a machine with a stable internet connection. If the training machines are separate, copy the finished folder to each of them.

1. Install the package with the data dependencies from the repository root:

   ```
   pip install -e ".[data]"
   ```

2. Print the plan. It lists the years, the number of channels, the amount of data to read, and the cache folder. It reads nothing from the source.

   ```
   python scripts/prepare_data.py plan --config era5_5p6deg
   ```

3. Download. Years that are already saved are skipped, so the command can be repeated after an interruption.

   ```
   python scripts/prepare_data.py download --config era5_5p6deg
   ```

4. Compute the statistics from the training years.

   ```
   python scripts/prepare_data.py statistics --config era5_5p6deg
   ```

5. Check that the copy is complete. The command exits with code 1 if anything is missing.

   ```
   python scripts/prepare_data.py check --config era5_5p6deg
   ```

## Script options

The script has no built-in dataset. Every command needs `--config`.

- `--config`: a path to a data config, or the name of a file in `configs/data` without `.yaml`.
- `--cache-root`: a folder to use instead of the `cache.root` of the config. A relative path is taken from the repository root. This is how a read-only copy, such as a Kaggle input folder, is used.
- `--years`: for `download` only. Downloads the listed years instead of every year in the splits. Each year must belong to a split of the config.

## Config fields

The code takes these values from the data config and does not assume them:

- `source.stored_levels`: the number of levels stored in each chunk of the source. It is used to estimate the amount of data to read.
- `window.step_hours`: the time between frames. It sets the expected number of steps per year and the continuity check.
- `transforms.precipitation_channels`: the channels that receive the log10(1000 x + 1) transform.
- `splits`: the years of each split. The statistics use the `train` split.

The source adapter still assumes the WeatherBench 2 coordinate names (`time`, `level`, `latitude`, `longitude`). A source with other names needs a new adapter.

## Auxiliary channels

Every frame receives 10 extra input channels that are not predicted:

- 3 static fields: land-sea mask, soil type, and surface geopotential (the last two standardised).
- 3 position terms: sin(latitude), cos(latitude) cos(longitude), and cos(latitude) sin(longitude).
- 4 time terms: sine and cosine of the year progress, and sine and cosine of the local day progress (which depends on longitude).

## Tests

The tests use small synthetic data and do not need the real download.

```
pip install -e ".[dev]"
pytest
```
