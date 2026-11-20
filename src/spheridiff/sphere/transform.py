import numpy as np
import torch
from torch import nn

from .harmonics import real_harmonics
from .quadrature import sphere_quadrature

ANALYSIS_METHODS = ("quadrature", "least_squares")


def quadrature_analysis(harmonics, weights):
    return weights[:, None] * harmonics


def least_squares_analysis(harmonics, weights):
    weighted = weights[:, None] * harmonics
    normal = harmonics.T @ weighted
    return np.linalg.solve(normal, weighted.T).T


def analysis_matrix(harmonics, weights, method):
    if method == "least_squares":
        return least_squares_analysis(harmonics, weights)
    return quadrature_analysis(harmonics, weights)


def check_degrees(grid, degrees, method):
    if method not in ANALYSIS_METHODS:
        raise ValueError(f"unknown analysis '{method}', choose from {ANALYSIS_METHODS}")
    if method == "quadrature" and degrees > grid.bandlimit:
        raise ValueError(f"quadrature supports at most {grid.bandlimit} degrees on this grid, not {degrees}")
    if degrees * degrees > grid.height * grid.width:
        raise ValueError(f"{degrees} degrees need more coefficients than the grid has points")


class SphericalTransform(nn.Module):
    def __init__(self, grid, degrees=None, analysis="quadrature"):
        super().__init__()
        self.degrees = degrees or grid.bandlimit
        check_degrees(grid, self.degrees, analysis)
        self.coefficients = self.degrees**2
        self.height = grid.height
        self.width = grid.width
        harmonics = real_harmonics(grid.latitudes, grid.longitudes, self.degrees)
        matrix = analysis_matrix(harmonics, sphere_quadrature(grid), analysis)
        self.register_buffer("analysis", torch.tensor(matrix, dtype=torch.float32), persistent=False)
        synthesis = torch.tensor(np.ascontiguousarray(harmonics.T), dtype=torch.float32)
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

    def analyse_points(self, values):
        return torch.matmul(self.analysis.mT, values)

    def synthesise_points(self, coefficients):
        return torch.matmul(self.synthesis.mT, coefficients)


def build_transform(grid, settings=None):
    settings = settings or {}
    return SphericalTransform(grid, settings.get("bandlimit"), settings.get("analysis", "quadrature"))
