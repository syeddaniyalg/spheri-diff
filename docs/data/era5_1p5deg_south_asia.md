# ERA5 at 1.5 Degrees over South Asia

This document describes the dataset defined by `configs/data/era5_1p5deg_south_asia.yaml`, where it comes from, and how to build the local copy.

## Source

The data is ERA5 reanalysis from WeatherBench 2, regridded to an equiangular grid with poles at 1.5 degrees.

- Store: `gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr`
- Grid of the store: 121 latitudes from -90 to 90 degrees and 240 longitudes from 0 to 358.5 degrees, both with a spacing of 1.5 degrees.
- Time: 6-hourly, from 1959-01-02 00:00 to 2021-12-31 18:00.

The store is public and needs no account. Every chunk holds a full global field for 8 time steps, so reading the region still transfers the whole globe. On the connection used to build the copy, one year took about 25 minutes.

## Selection

- Region: a patch of 16 x 16 points, latitudes 18.0 to 40.5 degrees north and longitudes 58.5 to 81.0 degrees east, which covers Pakistan and northern India.
- Channels, static fields and window: the same as in [era5_5p6deg.md](era5_5p6deg.md).
- Splits: train 2017 to 2019, validation 2020, test 2021. The training years are fewer than in the global data, because the download is slow and fine-tuning needs less data.
- Statistics: copied from the global 5.6 degree data, so that the patch is normalised on the same scale as the pre-training data.

## Local copy

The local copy is stored in `artifacts/data/era5_1p5deg_south_asia/`, which is not tracked by git. It has the same layout as the global copy and is about 0.3 GB.

```
python scripts/prepare_data.py download --config era5_1p5deg_south_asia
python scripts/prepare_data.py statistics --config era5_1p5deg_south_asia --statistics-from era5_5p6deg
python scripts/prepare_data.py check --config era5_1p5deg_south_asia
```
