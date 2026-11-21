from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .channels import channel_names
from .layout import CacheLayout
from .transforms import apply_precipitation_transform


@dataclass(frozen=True)
class Statistics:
    names: list
    mean: np.ndarray
    std: np.ndarray
    diff_std: np.ndarray

    def normalise(self, frames):
        frames -= self.mean[None, :, None, None]
        frames /= self.std[None, :, None, None]
        return frames

    def denormalise(self, frames):
        mean = torch.as_tensor(self.mean, device=frames.device, dtype=frames.dtype).view(-1, 1, 1)
        std = torch.as_tensor(self.std, device=frames.device, dtype=frames.dtype).view(-1, 1, 1)
        return frames * std + mean

    def loss_weights(self):
        weights = (self.std / self.diff_std) ** 2
        return weights / weights.mean()


class RunningMoments:
    def __init__(self, channels):
        self.count = 0
        self.total = np.zeros(channels)
        self.squares = np.zeros(channels)

    def update(self, frames):
        values = frames.astype(np.float64)
        self.count += values.shape[0] * values.shape[2] * values.shape[3]
        self.total += values.sum(axis=(0, 2, 3))
        self.squares += np.square(values).sum(axis=(0, 2, 3))

    @property
    def mean(self):
        return (self.total / self.count).astype(np.float32)

    @property
    def std(self):
        variance = self.squares / self.count - (self.total / self.count) ** 2
        return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def compute_statistics(layout, years, names, precipitation_channels, show_progress=False):
    values = RunningMoments(len(names))
    differences = RunningMoments(len(names))
    for year in tqdm(list(years), desc="statistics", disable=not show_progress):
        raw = np.load(layout.frames_path(year))
        frames = apply_precipitation_transform(raw, names, precipitation_channels)
        values.update(frames)
        differences.update(np.diff(frames, axis=0))
    return Statistics(names, values.mean, values.std, differences.std)


def save_statistics(statistics, path):
    np.savez(
        path,
        names=np.array(statistics.names),
        mean=statistics.mean,
        std=statistics.std,
        diff_std=statistics.diff_std,
    )


def load_statistics(path):
    data = np.load(path)
    names = [str(name) for name in data["names"]]
    return Statistics(names, data["mean"], data["std"], data["diff_std"])


def build_statistics(config):
    layout = CacheLayout(config.cache_root)
    first, last = config.splits["train"]
    names = channel_names(config.channels)
    years = range(first, last + 1)
    statistics = compute_statistics(
        layout, years, names, config.precipitation_channels, show_progress=True
    )
    save_statistics(statistics, layout.stats_path)
    return statistics


def copy_statistics(config, source_config):
    source = load_statistics(CacheLayout(source_config.cache_root).stats_path)
    if source.names != channel_names(config.channels):
        raise ValueError("the source statistics were built for a different channel set")
    save_statistics(source, CacheLayout(config.cache_root).stats_path)
    return source
