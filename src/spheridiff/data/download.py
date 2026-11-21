import calendar

import numpy as np
from tqdm import tqdm

from spheridiff.common.config import SECONDS_PER_DAY

from .layout import CacheLayout, write_array
from .source import (
    close_source,
    crop_region,
    load_static,
    open_source,
    select_year,
    stack_channels,
    year_seconds,
)


def all_years(config):
    years = set()
    for first, last in config.splits.values():
        years.update(range(first, last + 1))
    return sorted(years)


def select_years(config, requested=None):
    available = all_years(config)
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"years {unknown} are not in the splits of this config")
    return sorted(set(requested))


def expected_steps(year, step_seconds):
    days = 366 if calendar.isleap(year) else 365
    return days * SECONDS_PER_DAY // step_seconds


def check_year(frames, seconds, year, step_seconds):
    expected = expected_steps(year, step_seconds)
    if frames.shape[0] != expected:
        raise ValueError(f"{year}: expected {expected} steps, found {frames.shape[0]}")
    if not np.all(np.diff(seconds) == step_seconds):
        raise ValueError(f"{year}: time steps are not regular")
    if np.isnan(frames).any():
        raise ValueError(f"{year}: frames contain missing values")


def download_year(dataset, config, year, layout, show_progress=False):
    year_data = select_year(dataset, year)
    frames = stack_channels(year_data, config.channels, show_progress)
    seconds = year_seconds(year_data)
    check_year(frames, seconds, year, config.step_seconds)
    write_array(layout.frames_path(year), frames)
    write_array(layout.times_path(year), seconds)


def download_static(dataset, config, layout):
    static = load_static(dataset, config.channels.static_fields)
    np.savez(layout.static_path, **static)


def download_all(config, years=None):
    layout = CacheLayout(config.cache_root)
    selected = select_years(config, years)
    layout.create()
    source = open_source(config.source)
    dataset = crop_region(source, config.grid)
    try:
        if not layout.static_path.exists():
            download_static(dataset, config, layout)
            tqdm.write("saved static fields")
        pending = [year for year in selected if not layout.has_year(year)]
        tqdm.write(f"{len(pending)} years to download")
        progress = tqdm(pending, desc="years")
        for year in progress:
            progress.set_postfix(year=year)
            download_year(dataset, config, year, layout, show_progress=True)
    finally:
        close_source(source)
