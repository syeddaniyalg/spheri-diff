import numpy as np
import torch
from torch import nn

from .harmonics import coefficient_index
from .legendre import normalised_legendre
from .quadrature import latitude_weights


def order_scales(degrees):
    scales = np.full(degrees, np.sqrt(2.0))
    scales[0] = 1.0
    return scales


def check_longitudes(grid, degrees):
    expected = np.arange(grid.width) * 360.0 / grid.width
    if not np.allclose(grid.longitudes, expected):
        raise ValueError("the separable transform needs equally spaced longitudes that start at zero")
    if degrees > grid.width // 2:
        raise ValueError(f"{grid.width} longitudes support at most {grid.width // 2} degrees")


def legendre_by_order(grid, degrees):
    sines = np.sin(np.radians(grid.latitudes))
    return normalised_legendre(degrees, sines).transpose(1, 0, 2)


def analysis_weights(grid, legendre):
    scale = order_scales(legendre.shape[0])[:, None, None]
    area = latitude_weights(grid) * 2.0 * np.pi / grid.width
    return (scale * legendre * area[None, None, :]).transpose(0, 2, 1)


def synthesis_weights(legendre):
    halves = np.full(legendre.shape[0], 0.5)
    halves[0] = 1.0
    scale = (order_scales(legendre.shape[0]) * halves)[:, None, None]
    return scale * legendre


def stacked_position(degree, order, degrees):
    part = 1 if order < 0 else 0
    return part * degrees * degrees + abs(order) * degrees + degree


def gather_positions(degrees):
    positions = np.zeros(degrees * degrees, dtype=np.int64)
    for degree in range(degrees):
        for order in range(-degree, degree + 1):
            positions[coefficient_index(degree, order)] = stacked_position(degree, order, degrees)
    return positions


def scatter_positions(degrees):
    gather = gather_positions(degrees)
    positions = np.full(2 * degrees * degrees, degrees * degrees, dtype=np.int64)
    positions[gather] = np.arange(degrees * degrees)
    return positions


def as_buffer(values, dtype=torch.float32):
    return torch.tensor(np.ascontiguousarray(values), dtype=dtype)


class SeparableTransform(nn.Module):
    def __init__(self, grid, degrees=None):
        super().__init__()
        self.degrees = degrees or grid.bandlimit
        check_longitudes(grid, self.degrees)
        self.coefficients = self.degrees**2
        self.height = grid.height
        self.width = grid.width
        legendre = legendre_by_order(grid, self.degrees)
        self.register_buffer("forward_weights", as_buffer(analysis_weights(grid, legendre)), persistent=False)
        self.register_buffer("inverse_weights", as_buffer(synthesis_weights(legendre)), persistent=False)
        self.register_buffer("gather", as_buffer(gather_positions(self.degrees), torch.long), persistent=False)
        self.register_buffer("scatter", as_buffer(scatter_positions(self.degrees), torch.long), persistent=False)

    def analyse_points(self, values):
        fields = values.unflatten(-2, (self.height, self.width))
        spectrum = torch.fft.rfft(fields, dim=-2)[..., : self.degrees, :]
        parts = torch.stack([spectrum.real, -spectrum.imag], dim=-4)
        stacked = torch.einsum("...shmc,mhl->...smlc", parts, self.forward_weights)
        return stacked.flatten(-4, -2).index_select(-2, self.gather)

    def synthesise_points(self, coefficients):
        padding = coefficients.new_zeros(*coefficients.shape[:-2], 1, coefficients.shape[-1])
        padded = torch.cat([coefficients, padding], dim=-2)
        stacked = padded.index_select(-2, self.scatter).unflatten(-2, (2, self.degrees, self.degrees))
        parts = torch.einsum("...smlc,mlh->...shmc", stacked, self.inverse_weights)
        spectrum = torch.complex(parts[..., 0, :, :, :], -parts[..., 1, :, :, :])
        fields = torch.fft.irfft(spectrum, n=self.width, dim=-2, norm="forward")
        return fields.flatten(-3, -2)
