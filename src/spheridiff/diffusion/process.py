from dataclasses import dataclass

import torch

from spheridiff.sphere.transform import SphericalTransform

from .noise import GaussianNoise, SpectralNoise
from .schedule import RectifiedSchedule


class IdentityProjection:
    def __call__(self, clean):
        return clean


class BandLimitProjection:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, clean):
        return self.transform.to(clean.device).band_limit(clean)


def expand_levels(values, like):
    while values.ndim < like.ndim:
        values = values[..., None]
    return values


@dataclass
class DiffusionProcess:
    schedule: RectifiedSchedule
    noise: object
    projection: object

    def sample_levels(self, batch, frames, device, generator=None):
        return torch.rand(batch, frames, device=device, generator=generator)

    def coefficients(self, levels, like):
        alpha, sigma = self.schedule(levels)
        return expand_levels(alpha, like), expand_levels(sigma, like)

    def corrupt(self, clean, levels, generator=None):
        target = self.projection(clean)
        noise = self.noise.sample(clean.shape, clean.device, generator)
        alpha, sigma = self.coefficients(levels, clean)
        return alpha * target + sigma * noise, target


def build_process(noise, projection, grid):
    needs_transform = noise == "spectral" or projection == "band_limit"
    transform = SphericalTransform(grid) if needs_transform else None
    noise_source = SpectralNoise(transform) if noise == "spectral" else GaussianNoise()
    projector = IdentityProjection() if projection == "identity" else BandLimitProjection(transform)
    return DiffusionProcess(RectifiedSchedule(), noise_source, projector)
