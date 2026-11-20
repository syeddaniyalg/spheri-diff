import numpy as np


def clenshaw_curtis_weights(points):
    intervals = points - 1
    nodes = np.arange(points)
    orders = np.arange(1, intervals // 2 + 1)
    edge = np.where((nodes == 0) | (nodes == intervals), 1.0, 2.0)
    scale = np.where(2 * orders == intervals, 1.0, 2.0) / (4 * orders**2 - 1)
    cosines = np.cos(2.0 * np.pi * np.outer(orders, nodes) / intervals)
    return edge / intervals * (1.0 - scale @ cosines)


def gauss_legendre_weights(latitudes):
    sines = np.sin(np.radians(latitudes))
    _, weights = np.polynomial.legendre.leggauss(len(latitudes))
    ranks = np.argsort(np.argsort(sines))
    return weights[ranks]


def latitude_weights(grid):
    if grid.quadrature == "gauss_legendre":
        return gauss_legendre_weights(grid.latitudes)
    return clenshaw_curtis_weights(grid.height)


def sphere_quadrature(grid):
    longitude_weight = 2.0 * np.pi / grid.width
    return np.repeat(latitude_weights(grid), grid.width) * longitude_weight
