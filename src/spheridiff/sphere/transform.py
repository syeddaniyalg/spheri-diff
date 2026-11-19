import numpy as np
import torch
from torch import nn

from .harmonics import real_harmonics
from .quadrature import sphere_quadrature


class SphericalTransform(nn.Module):
    def __init__(self, grid, degrees=None):
        super().__init__()
        self.degrees = degrees or grid.bandlimit
        self.coefficients = self.degrees**2
        self.height = grid.height
        self.width = grid.width
        harmonics = real_harmonics(grid.latitudes, grid.longitudes, self.degrees)
        weights = sphere_quadrature(grid)
        analysis = torch.tensor(weights[:, None] * harmonics, dtype=torch.float32)
        synthesis = torch.tensor(np.ascontiguousarray(harmonics.T), dtype=torch.float32)
        self.register_buffer("analysis", analysis, persistent=False)
        self.register_buffer("synthesis", synthesis, persistent=False)

    def analyse(self, field):
        with torch.autocast(device_type=field.device.type, enabled=False):
            return field.float().flatten(-2) @ self.analysis

    def synthesise(self, coefficients):
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            flat = coefficients.float() @ self.synthesis
            return flat.unflatten(-1, (self.height, self.width))

    def synthesis_adjoint(self, field):
        with torch.autocast(device_type=field.device.type, enabled=False):
            return field.float().flatten(-2) @ self.synthesis.T

    def band_limit(self, field):
        return self.synthesise(self.analyse(field))
