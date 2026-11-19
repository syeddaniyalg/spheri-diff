import math

import torch


class GaussianNoise:
    def sample(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)


class SpectralNoise:
    def __init__(self, transform):
        self.transform = transform

    def sample(self, shape, device, generator=None):
        transform = self.transform.to(device)
        coefficients = torch.randn(
            *shape[:-2], transform.coefficients, device=device, generator=generator
        )
        scale = math.sqrt(4.0 * math.pi / transform.coefficients)
        return transform.synthesise(coefficients * scale)
