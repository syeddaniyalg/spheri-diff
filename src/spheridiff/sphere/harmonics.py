import numpy as np

from .legendre import normalised_legendre


def coefficient_index(degree, order):
    return degree * degree + degree + order


def fill_degree(matrix, degree, legendre, phi):
    matrix[:, :, coefficient_index(degree, 0)] = legendre[degree, 0][:, None]
    for order in range(1, degree + 1):
        scaled = np.sqrt(2.0) * legendre[degree, order][:, None]
        matrix[:, :, coefficient_index(degree, order)] = scaled * np.cos(order * phi)[None, :]
        matrix[:, :, coefficient_index(degree, -order)] = scaled * np.sin(order * phi)[None, :]


def harmonics_of_degree(legendre, phi, degree):
    columns = [np.sqrt(2.0) * legendre[degree, order][:, None] * np.sin(order * phi)[None, :] for order in range(degree, 0, -1)]
    columns.append(np.broadcast_to(legendre[degree, 0][:, None], (legendre.shape[-1], len(phi))))
    columns += [np.sqrt(2.0) * legendre[degree, order][:, None] * np.cos(order * phi)[None, :] for order in range(1, degree + 1)]
    return np.stack(columns, axis=-1).reshape(-1, 2 * degree + 1)


def real_harmonics(latitudes, longitudes, degrees):
    sines = np.sin(np.radians(latitudes))
    legendre = normalised_legendre(degrees, sines)
    phi = np.radians(longitudes)
    matrix = np.zeros((len(latitudes), len(longitudes), degrees * degrees))
    for degree in range(degrees):
        fill_degree(matrix, degree, legendre, phi)
    return matrix.reshape(-1, degrees * degrees)
