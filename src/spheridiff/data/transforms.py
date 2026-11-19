import numpy as np


def precipitation_indices(names, precipitation_channels):
    return [names.index(name) for name in precipitation_channels if name in names]


def apply_precipitation_transform(frames, names, precipitation_channels):
    indices = precipitation_indices(names, precipitation_channels)
    if indices:
        frames[:, indices] = np.log10(frames[:, indices] * 1000.0 + 1.0)
    return frames


def undo_precipitation_transform(frames, names, precipitation_channels):
    result = frames.clone()
    indices = precipitation_indices(names, precipitation_channels)
    if indices:
        result[..., indices, :, :] = (10.0 ** result[..., indices, :, :] - 1.0) / 1000.0
    return result
