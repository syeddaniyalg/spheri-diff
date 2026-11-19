import numpy as np


def clenshaw_curtis_weights(points):
    intervals = points - 1
    nodes = np.arange(points)
    orders = np.arange(1, intervals // 2 + 1)
    edge = np.where((nodes == 0) | (nodes == intervals), 1.0, 2.0)
    scale = np.where(2 * orders == intervals, 1.0, 2.0) / (4 * orders**2 - 1)
    cosines = np.cos(2.0 * np.pi * np.outer(orders, nodes) / intervals)
    return edge / intervals * (1.0 - scale @ cosines)


def sphere_quadrature(grid):
    latitude_weights = clenshaw_curtis_weights(grid.height)
    longitude_weight = 2.0 * np.pi / grid.width
    return np.repeat(latitude_weights, grid.width) * longitude_weight
