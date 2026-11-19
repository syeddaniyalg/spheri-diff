import math

import numpy as np
import torch
import torch.nn.functional as functional


class SpatialAverage:
    def __init__(self, factor):
        self.factor = factor

    def apply(self, field):
        batch, frames, channels, height, width = field.shape
        merged = field.reshape(batch * frames * channels, 1, height, width)
        pooled = functional.avg_pool2d(merged, self.factor)
        return pooled.reshape(batch, frames, channels, *pooled.shape[-2:])

    def adjoint(self, observation):
        rows = observation.repeat_interleave(self.factor, dim=-2)
        return rows.repeat_interleave(self.factor, dim=-1) / self.factor**2

    def baseline(self, observation):
        return self.adjoint(observation) * self.factor**2


class TemporalAverage:
    def __init__(self, frames):
        self.frames = frames

    def apply(self, field):
        return field.mean(dim=1, keepdim=True)

    def adjoint(self, observation):
        return observation.expand(-1, self.frames, -1, -1, -1) / self.frames

    def baseline(self, observation):
        return observation.expand(-1, self.frames, -1, -1, -1)


class SpatioTemporalAverage:
    def __init__(self, factor, frames):
        self.spatial = SpatialAverage(factor)
        self.temporal = TemporalAverage(frames)

    def apply(self, field):
        return self.temporal.apply(self.spatial.apply(field))

    def adjoint(self, observation):
        return self.spatial.adjoint(self.temporal.adjoint(observation))

    def baseline(self, observation):
        return self.spatial.baseline(self.temporal.baseline(observation))


class SparseMask:
    def __init__(self, mask):
        self.mask = mask

    def apply(self, field):
        return field * self.mask

    def adjoint(self, observation):
        return observation * self.mask

    def baseline(self, observation):
        return observation


def interpolation_weights(frames, observed):
    weights = torch.zeros(frames, len(observed))
    for column in range(len(observed)):
        basis = [1.0 if index == column else 0.0 for index in range(len(observed))]
        for frame in range(frames):
            weights[frame, column] = float(np.interp(frame, observed, basis))
    return weights


class FrameMask(SparseMask):
    def __init__(self, frames, observed, device):
        mask = torch.zeros(1, frames, 1, 1, 1)
        mask[:, list(observed)] = 1.0
        super().__init__(mask.to(device))
        self.frames = frames
        self.observed = sorted(observed)

    def baseline(self, observation):
        weights = interpolation_weights(self.frames, self.observed).to(observation)
        return torch.einsum("tk,bkchw->btchw", weights, observation[:, self.observed])


def random_space_time_mask(frames, height, width, fraction, device, seed):
    generator = torch.Generator().manual_seed(seed)
    total = frames * height * width
    chosen = torch.randperm(total, generator=generator)[: max(1, math.ceil(fraction * total))]
    mask = torch.zeros(total)
    mask[chosen] = 1.0
    return mask.reshape(1, frames, 1, height, width).to(device)


def random_sensor_mask(height, width, fraction, device, seed):
    generator = torch.Generator().manual_seed(seed)
    count = max(1, math.ceil(fraction * height * width))
    chosen = torch.randperm(height * width, generator=generator)[:count]
    mask = torch.zeros(height * width)
    mask[chosen] = 1.0
    return mask.reshape(1, 1, 1, height, width).to(device)
