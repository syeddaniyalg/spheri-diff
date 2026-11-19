import math

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


class SparseMask:
    def __init__(self, mask):
        self.mask = mask

    def apply(self, field):
        return field * self.mask

    def adjoint(self, observation):
        return observation * self.mask

    def baseline(self, observation):
        return observation


def random_sensor_mask(height, width, fraction, device, seed):
    generator = torch.Generator().manual_seed(seed)
    count = max(1, math.ceil(fraction * height * width))
    chosen = torch.randperm(height * width, generator=generator)[:count]
    mask = torch.zeros(height * width)
    mask[chosen] = 1.0
    return mask.reshape(1, 1, 1, height, width).to(device)
