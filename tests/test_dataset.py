import numpy as np
import pytest

from helpers import make_cache, make_config
from spheridiff.data.batches import random_batches
from spheridiff.data.channels import channel_names
from spheridiff.data.layout import CacheLayout
from spheridiff.data.loading import build_dataset, check_continuity, load_dataset_statistics
from spheridiff.data.plan import check_cache
from spheridiff.data.statistics import build_statistics


@pytest.fixture
def config(tmp_path):
    config = make_config(tmp_path / "cache")
    make_cache(config)
    build_statistics(config)
    return config


def test_cache_is_reported_complete(config):
    assert check_cache(config) == []


def test_statistics_match_training_data(config):
    statistics = load_dataset_statistics(config)
    assert statistics.names == channel_names(config.channels)
    assert statistics.mean.shape == (3,)
    assert np.all(statistics.std > 0)
    assert np.isclose(statistics.loss_weights().mean(), 1.0)


def test_training_frames_are_normalised(config):
    dataset = build_dataset(config, "train")
    assert np.allclose(dataset.frames.mean(axis=(0, 2, 3)), 0.0, atol=1e-4)
    assert np.allclose(dataset.frames.std(axis=(0, 2, 3)), 1.0, atol=1e-3)


def test_window_shapes(config):
    sample = build_dataset(config, "train")[0]
    assert sample["frames"].shape == (5, 3, 8, 16)
    assert sample["auxiliary"].shape == (5, 10, 8, 16)
    assert sample["seconds"].shape == (5,)


def test_window_count(config):
    dataset = build_dataset(config, "val")
    assert len(dataset) == dataset.frames.shape[0] - 4


def test_random_batches_repeat_with_same_seed(config):
    dataset = build_dataset(config, "train")
    first = next(random_batches(dataset, 4, seed=7))
    second = next(random_batches(dataset, 4, seed=7))
    assert np.array_equal(first["frames"].numpy(), second["frames"].numpy())
    assert first["frames"].shape == (4, 5, 3, 8, 16)


def test_gap_in_time_is_rejected():
    seconds = np.array([0, 21600, 50000], dtype=np.int64)
    with pytest.raises(ValueError):
        check_continuity(seconds, 21600)


def test_missing_year_is_reported(config):
    layout = CacheLayout(config.cache_root)
    layout.frames_path(2011).unlink()
    assert "2011 is missing" in check_cache(config)
