import numpy as np
from tqdm import tqdm

from spheridiff.grid.grid import GridSpec

from .auxiliary import static_auxiliary
from .channels import channel_names
from .dataset import WeatherWindows
from .layout import CacheLayout
from .statistics import load_statistics
from .transforms import apply_precipitation_transform


def split_years(config, split):
    first, last = config.splits[split]
    return list(range(first, last + 1))


def load_grid(config):
    layout = CacheLayout(config.cache_root)
    static = np.load(layout.static_path)
    return GridSpec(static["latitudes"], static["longitudes"])


def load_dataset_statistics(config):
    layout = CacheLayout(config.cache_root)
    statistics = load_statistics(layout.stats_path)
    if statistics.names != channel_names(config.channels):
        raise ValueError("statistics were built for a different channel set")
    return statistics


def check_continuity(seconds, step_seconds):
    if not np.all(np.diff(seconds) == step_seconds):
        raise ValueError(f"the selected years are not one continuous series with {step_seconds} s steps")


def load_split_frames(config, layout, years, names, statistics, show_progress=False):
    shapes = [np.load(layout.frames_path(year), mmap_mode="r").shape for year in years]
    total = sum(shape[0] for shape in shapes)
    frames = np.empty((total, *shapes[0][1:]), dtype=np.float32)
    seconds = np.empty(total, dtype=np.int64)
    start = 0
    for year, shape in tqdm(list(zip(years, shapes)), desc="loading", disable=not show_progress):
        stop = start + shape[0]
        raw = np.load(layout.frames_path(year))
        block = apply_precipitation_transform(raw, names, config.precipitation_channels)
        frames[start:stop] = statistics.normalise(block)
        seconds[start:stop] = np.load(layout.times_path(year))
        start = stop
    check_continuity(seconds, config.step_seconds)
    return frames, seconds


def build_dataset(config, split, show_progress=False, window_frames=None):
    layout = CacheLayout(config.cache_root)
    names = channel_names(config.channels)
    statistics = load_dataset_statistics(config)
    years = split_years(config, split)
    frames, seconds = load_split_frames(config, layout, years, names, statistics, show_progress)
    static = np.load(layout.static_path)
    static_maps = static_auxiliary(static, config.channels.static_fields)
    count = window_frames or config.frames
    return WeatherWindows(frames, seconds, static["longitudes"], static_maps, count)
