import math

import numpy as np
import torch
from torch import nn

from spheridiff.models.layers.norm import RMSNorm
from spheridiff.sphere.quadrature import sphere_quadrature
from spheridiff.sphere.slepian import patch_weights

BASIS_FUNCTIONS = 4
RING_SPACING_DEGREES = 180.0 / 31.0


def unit_vectors(latitudes, longitudes):
    phi, lam = np.radians(latitudes), np.radians(longitudes)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


def local_frames(latitudes, longitudes):
    phi, lam = np.radians(latitudes), np.radians(longitudes)
    east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], axis=-1)
    north = np.stack([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)], axis=-1)
    return east, north


def grid_points(grid):
    latitudes, longitudes = np.meshgrid(grid.latitudes, grid.longitudes, indexing="ij")
    return latitudes.ravel(), longitudes.ravel()


def distances_and_directions(latitudes, longitudes):
    points = unit_vectors(latitudes, longitudes)
    east, north = local_frames(latitudes, longitudes)
    cosines = np.clip(points @ points.T, -1.0, 1.0)
    tangent = points[None, :, :] - cosines[:, :, None] * points[:, None, :]
    directions = np.arctan2(np.einsum("ojc,oc->oj", tangent, east), np.einsum("ojc,oc->oj", tangent, north))
    return np.arccos(cosines), directions


def hat(distance, centre, spacing):
    return np.clip(1.0 - np.abs(distance - centre) / spacing, 0.0, None)


def basis_values(distance, direction, spacing):
    centre, ring = hat(distance, 0.0, spacing), hat(distance, spacing, spacing)
    ring = np.where(distance < 2.0 * spacing, ring, 0.0)
    return np.stack([centre, ring, ring * np.cos(direction), ring * np.sin(direction)])


def continuous_norms(spacing, samples=4000):
    distance = np.linspace(0.0, 2.0 * spacing, samples)
    area = np.sin(distance)
    radial = [hat(distance, 0.0, spacing), hat(distance, spacing, spacing)]
    integrals = [np.trapezoid(profile * area, distance) for profile in radial]
    return np.array([2.0 * math.pi * integrals[0], 2.0 * math.pi * integrals[1], 4.0 * integrals[1], 4.0 * integrals[1]])


def point_weights(grid, regional):
    return patch_weights(grid) if regional else sphere_quadrature(grid)


def kernel_matrix(grid, regional, spacing_degrees=RING_SPACING_DEGREES):
    spacing = math.radians(spacing_degrees)
    latitudes, longitudes = grid_points(grid)
    distance, direction = distances_and_directions(latitudes, longitudes)
    values = basis_values(distance, direction, spacing)
    values = values / continuous_norms(spacing)[:, None, None] * point_weights(grid, regional)[None, None, :]
    return values.reshape(BASIS_FUNCTIONS * len(latitudes), len(latitudes))


class SphericalConvolution(nn.Module):
    def __init__(self, grid, width, regional=False, spacing_degrees=RING_SPACING_DEGREES):
        super().__init__()
        matrix = torch.tensor(kernel_matrix(grid, regional, spacing_degrees), dtype=torch.float32).to_sparse_csr()
        self.points = grid.height * grid.width
        self.register_buffer("rows", matrix.crow_indices(), persistent=False)
        self.register_buffer("columns", matrix.col_indices(), persistent=False)
        self.register_buffer("entries", matrix.values(), persistent=False)
        self.weight = nn.Parameter(torch.zeros(width, width, BASIS_FUNCTIONS))

    def matrix(self, dtype):
        shape = (BASIS_FUNCTIONS * self.points, self.points)
        return torch.sparse_csr_tensor(self.rows, self.columns, self.entries.to(dtype), size=shape)

    def forward(self, values):
        batch, frames, points, width = values.shape
        flat = values.permute(2, 0, 1, 3).reshape(points, -1)
        gathered = (self.matrix(flat.dtype) @ flat).reshape(BASIS_FUNCTIONS, points, batch, frames, width)
        mixed = torch.einsum("knbtc,ock->btno", gathered, self.weight.to(flat.dtype))
        return mixed


class LocalStage(nn.Module):
    def __init__(self, grid, width, regional=False):
        super().__init__()
        self.norm = RMSNorm(width)
        self.convolution = SphericalConvolution(grid, width, regional)

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            normed = self.norm(values).to(values.dtype)
            return self.convolution(normed)
