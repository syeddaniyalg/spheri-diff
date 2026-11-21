import math

import numpy as np
import torch

from .harmonics import real_harmonics
from .transform import MatrixTransform

QUADRATURE_POINTS = 64
ROWS_PER_CHUNK = 8


def cell_bounds(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    spacing = np.diff(ordered).mean()
    return ordered[0] - spacing / 2.0, ordered[-1] + spacing / 2.0


def region_area(latitude_bounds, longitude_bounds):
    width = np.radians(longitude_bounds[1] - longitude_bounds[0])
    south, north = np.radians(latitude_bounds)
    return width * (np.sin(north) - np.sin(south))


def shannon_number(area, degrees):
    return math.ceil(round(area / (4.0 * math.pi) * degrees**2, 6))


def gauss_nodes(bounds, points):
    nodes, weights = np.polynomial.legendre.leggauss(points)
    low, high = bounds
    half = (high - low) / 2.0
    return low + half * (nodes + 1.0), weights * np.radians(half)


def concentration_matrix(latitude_bounds, longitude_bounds, degrees, points=QUADRATURE_POINTS):
    latitudes, latitude_weights = gauss_nodes(latitude_bounds, points)
    longitudes, longitude_weights = gauss_nodes(longitude_bounds, points)
    matrix = np.zeros((degrees * degrees, degrees * degrees))
    for start in range(0, points, ROWS_PER_CHUNK):
        rows = slice(start, start + ROWS_PER_CHUNK)
        harmonics = real_harmonics(latitudes[rows], longitudes, degrees)
        area = np.cos(np.radians(latitudes[rows]))[:, None] * latitude_weights[rows, None] * longitude_weights[None, :]
        matrix += harmonics.T @ (area.reshape(-1, 1) * harmonics)
    return matrix


def slepian_basis(matrix, count):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1][:count]
    return eigenvalues[order], eigenvectors[:, order]


def patch_weights(grid):
    spacing = np.radians(np.diff(np.sort(grid.latitudes)).mean())
    width = np.radians(np.diff(np.sort(grid.longitudes)).mean())
    rows = np.cos(np.radians(grid.latitudes)) * spacing * width
    return np.repeat(rows, grid.width)


def least_squares(functions, weights):
    weighted = weights[:, None] * functions
    return np.linalg.solve(functions.T @ weighted, weighted.T).T


class SlepianTransform(MatrixTransform):
    def __init__(self, grid, degrees, points=QUADRATURE_POINTS):
        latitude_bounds, longitude_bounds = cell_bounds(grid.latitudes), cell_bounds(grid.longitudes)
        self.area = region_area(latitude_bounds, longitude_bounds)
        count = shannon_number(self.area, degrees)
        if count > grid.height * grid.width:
            raise ValueError(f"{count} Slepian functions need more points than the patch has")
        matrix = concentration_matrix(latitude_bounds, longitude_bounds, degrees, points)
        self.eigenvalues, basis = slepian_basis(matrix, count)
        functions = real_harmonics(grid.latitudes, grid.longitudes, degrees) @ basis
        super().__init__(least_squares(functions, patch_weights(grid)), functions.T, grid, self.area / count)
        self.degrees = degrees
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32), persistent=False)
