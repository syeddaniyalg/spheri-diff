import math
from functools import lru_cache

import numpy as np
import torch

from .harmonics import harmonics_of_degree, real_harmonics
from .legendre import normalised_legendre
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
        self.regional = True
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32), persistent=False)

    def harmonic_basis(self, count):
        return self.basis[:count]


def unit_vectors(latitudes, longitudes):
    phi, lam = np.radians(latitudes), np.radians(longitudes)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


def grid_vectors(latitudes, longitudes):
    lat, lon = np.meshgrid(latitudes, longitudes, indexing="ij")
    return unit_vectors(lat.ravel(), lon.ravel())


def reproducing_kernel(first, second, degrees):
    cosines = np.clip(first @ second.T, -1.0, 1.0)
    previous, current = np.ones_like(cosines), cosines.copy()
    total = (previous + 3.0 * current) / (4.0 * math.pi)
    for degree in range(2, degrees):
        previous, current = current, ((2 * degree - 1) * cosines * current - (degree - 1) * previous) / degree
        total += current * ((2 * degree + 1) / (4.0 * math.pi))
    return total


def box_quadrature(latitude_bounds, longitude_bounds, points):
    latitudes, latitude_weights = gauss_nodes(latitude_bounds, points)
    longitudes, longitude_weights = gauss_nodes(longitude_bounds, points)
    area = (np.cos(np.radians(latitudes)) * latitude_weights)[:, None] * longitude_weights[None, :]
    return latitudes, longitudes, area.ravel()


@lru_cache(maxsize=16)
def kernel_slepian(latitude_bounds, longitude_bounds, degrees, count, points):
    latitudes, longitudes, area = box_quadrature(latitude_bounds, longitude_bounds, points)
    nodes = grid_vectors(latitudes, longitudes)
    root = np.sqrt(area)
    gram = root[:, None] * reproducing_kernel(nodes, nodes, degrees) * root[None, :]
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:count]
    weights = root[:, None] * vectors[:, order] / np.sqrt(eigenvalues[order])
    return eigenvalues[order], latitudes, longitudes, nodes, weights


class KernelSlepianTransform(MatrixTransform):
    def __init__(self, grid, degrees, bounds=None, points=QUADRATURE_POINTS):
        latitude_bounds, longitude_bounds = bounds or (cell_bounds(grid.latitudes), cell_bounds(grid.longitudes))
        self.area = region_area(latitude_bounds, longitude_bounds)
        count = shannon_number(self.area, degrees)
        if count > grid.height * grid.width:
            raise ValueError(f"{count} Slepian functions need more points than the grid has")
        self.bandlimit = degrees
        key = (tuple(map(float, latitude_bounds)), tuple(map(float, longitude_bounds)), degrees, count, points)
        self.eigenvalues, self.quadrature_latitudes, self.quadrature_longitudes, self.nodes, self.weights = kernel_slepian(*key)
        functions = self.functions_at(grid.latitudes, grid.longitudes)
        super().__init__(least_squares(functions, patch_weights(grid)), functions.T, grid, self.area / count)
        self.degrees = degrees
        self.regional = True

    def functions_at(self, latitudes, longitudes):
        return reproducing_kernel(grid_vectors(latitudes, longitudes), self.nodes, self.bandlimit) @ self.weights

    def harmonic_basis(self, count):
        degrees = math.isqrt(count)
        harmonics = real_harmonics(self.quadrature_latitudes, self.quadrature_longitudes, degrees)
        return torch.tensor(harmonics.T @ self.weights, dtype=torch.float32)

    def degree_blocks(self):
        legendre = normalised_legendre(self.degrees, np.sin(np.radians(self.quadrature_latitudes)))
        phi = np.radians(self.quadrature_longitudes)
        for degree in range(self.degrees):
            yield harmonics_of_degree(legendre, phi, degree).T @ self.weights
