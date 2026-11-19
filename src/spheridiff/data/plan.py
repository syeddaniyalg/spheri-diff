import numpy as np

from spheridiff.common.config import SECONDS_PER_DAY

from .channels import channel_names
from .download import all_years
from .layout import CacheLayout

DAYS_PER_YEAR = 365.25
BYTES_PER_VALUE = 4
GIGABYTE = 1024**3


def steps_per_year(config):
    return round(DAYS_PER_YEAR * SECONDS_PER_DAY / config.step_seconds)


def read_bytes_per_year(config):
    channels = config.channels
    pressure = len(channels.pressure_variables) * config.source.stored_levels
    variables = pressure + len(channels.surface_variables)
    points = config.grid.height * config.grid.width
    return variables * points * BYTES_PER_VALUE * steps_per_year(config)


def cache_bytes_per_year(config):
    points = config.grid.height * config.grid.width
    channels = len(channel_names(config.channels))
    return channels * points * BYTES_PER_VALUE * steps_per_year(config)


def plan_lines(config):
    layout = CacheLayout(config.cache_root)
    years = all_years(config)
    missing = [year for year in years if not layout.has_year(year)]
    read_size = read_bytes_per_year(config) * len(missing) / GIGABYTE
    cache_size = cache_bytes_per_year(config) * len(years) / GIGABYTE
    return [
        f"source: {config.source.url}",
        f"cache folder: {layout.root}",
        f"channels: {len(channel_names(config.channels))}",
        f"years: {years[0]} to {years[-1]} ({len(years)} years)",
        f"years still missing: {missing if missing else 'none'}",
        f"data to read from the source: about {read_size:.1f} GB",
        f"size of the finished cache: about {cache_size:.1f} GB",
    ]


def check_cache(config):
    layout = CacheLayout(config.cache_root)
    names = channel_names(config.channels)
    problems = []
    if not layout.static_path.exists():
        problems.append("static fields are missing")
    if not layout.stats_path.exists():
        problems.append("statistics are missing")
    for year in all_years(config):
        if not layout.has_year(year):
            problems.append(f"{year} is missing")
            continue
        shape = np.load(layout.frames_path(year), mmap_mode="r").shape
        if shape[1] != len(names) or shape[2:] != (config.grid.height, config.grid.width):
            problems.append(f"{year} has the wrong shape {shape}")
    return problems
