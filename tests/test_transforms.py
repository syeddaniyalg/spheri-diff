import numpy as np
import torch

from spheridiff.data.transforms import (
    apply_precipitation_transform,
    undo_precipitation_transform,
)

PRECIPITATION = "total_precipitation_6hr"
NAMES = ["500_temperature", PRECIPITATION]
PRECIPITATION_CHANNELS = (PRECIPITATION,)


def test_precipitation_round_trip():
    generator = np.random.default_rng(0)
    raw = generator.random((3, 2, 4, 8)).astype(np.float32) * 0.01
    transformed = apply_precipitation_transform(raw.copy(), NAMES, PRECIPITATION_CHANNELS)
    restored = undo_precipitation_transform(
        torch.from_numpy(transformed), NAMES, PRECIPITATION_CHANNELS
    )
    assert np.allclose(restored.numpy(), raw, atol=1e-6)


def test_other_channels_are_unchanged():
    raw = np.ones((2, 2, 4, 8), dtype=np.float32)
    transformed = apply_precipitation_transform(raw.copy(), NAMES, PRECIPITATION_CHANNELS)
    assert np.array_equal(transformed[:, 0], raw[:, 0])


def test_missing_precipitation_leaves_frames_unchanged():
    raw = np.ones((2, 1, 4, 8), dtype=np.float32)
    assert np.array_equal(
        apply_precipitation_transform(raw.copy(), ["a"], PRECIPITATION_CHANNELS), raw
    )


def test_no_precipitation_channels_leaves_frames_unchanged():
    raw = np.ones((2, 2, 4, 8), dtype=np.float32)
    assert np.array_equal(apply_precipitation_transform(raw.copy(), NAMES, ()), raw)


def test_other_precipitation_name_is_transformed():
    names = ["total_precipitation_24hr"]
    raw = np.full((2, 1, 4, 8), 0.001, dtype=np.float32)
    transformed = apply_precipitation_transform(raw.copy(), names, tuple(names))
    assert np.allclose(transformed, np.log10(2.0))
