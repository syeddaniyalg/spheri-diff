import numpy as np

from spheridiff.common.config import ChannelConfig, DataConfig, GridConfig, SourceConfig
from spheridiff.data.channels import channel_names
from spheridiff.data.download import expected_steps
from spheridiff.data.layout import CacheLayout, write_array
from spheridiff.grid.grid import GridSpec

HEIGHT, WIDTH = 8, 16
FIRST_YEAR = 2010
SECONDS_2010 = int(np.datetime64("2010-01-01T00:00", "s").astype(np.int64))


def randomise_weights(model, deviation=0.2):
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            parameter.data.normal_(0.0, deviation)
    return model


def make_grid(height=32, width=64):
    latitudes = np.linspace(-90.0, 90.0, height)
    longitudes = np.arange(width) * 360.0 / width
    return GridSpec(latitudes, longitudes)


def make_config(root, years=(2010, 2011), step_hours=6):
    channels = ChannelConfig(
        pressure_variables=("temperature",),
        levels=(500, 850),
        surface_variables=("total_precipitation_6hr",),
        static_fields=("land_sea_mask", "soil_type", "geopotential_at_surface"),
    )
    return DataConfig(
        source=SourceConfig(url="unused", storage_options={}, stored_levels=13),
        grid=GridConfig(height=HEIGHT, width=WIDTH),
        channels=channels,
        frames=5,
        step_hours=step_hours,
        precipitation_channels=("total_precipitation_6hr",),
        splits={"train": (years[0], years[0]), "val": (years[1], years[1])},
        cache_root=root,
    )


def make_cache(config, seed=0):
    generator = np.random.default_rng(seed)
    layout = CacheLayout(config.cache_root)
    layout.create()
    names = channel_names(config.channels)
    start = SECONDS_2010
    step = config.step_seconds
    for year in sorted({y for pair in config.splits.values() for y in pair}):
        steps = expected_steps(year, step)
        frames = generator.random((steps, len(names), HEIGHT, WIDTH)).astype(np.float32)
        frames[:, -1] *= 0.01
        write_array(layout.frames_path(year), frames)
        write_array(layout.times_path(year), start + step * np.arange(steps, dtype=np.int64))
        start += step * steps
    static = {
        "latitudes": np.linspace(-90.0, 90.0, HEIGHT),
        "longitudes": np.arange(WIDTH) * 360.0 / WIDTH,
        "land_sea_mask": generator.random((HEIGHT, WIDTH)).astype(np.float32),
        "soil_type": generator.integers(0, 7, (HEIGHT, WIDTH)).astype(np.float32),
        "geopotential_at_surface": generator.random((HEIGHT, WIDTH)).astype(np.float32),
    }
    np.savez(layout.static_path, **static)
    return layout
