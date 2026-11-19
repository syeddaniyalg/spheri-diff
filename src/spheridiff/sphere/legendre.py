import numpy as np


def diagonal_terms(values, sines):
    for order in range(1, values.shape[0]):
        factor = np.sqrt(1.0 + 0.5 / order)
        values[order, order] = factor * sines * values[order - 1, order - 1]


def first_off_diagonal_terms(values, points):
    for order in range(values.shape[0] - 1):
        values[order + 1, order] = np.sqrt(2.0 * order + 3.0) * points * values[order, order]


def recurrence_terms(values, points):
    degrees = values.shape[0]
    for order in range(degrees):
        for degree in range(order + 2, degrees):
            first = np.sqrt((4.0 * degree**2 - 1.0) / (degree**2 - order**2))
            second = np.sqrt(
                ((degree - 1.0) ** 2 - order**2) / (4.0 * (degree - 1.0) ** 2 - 1.0)
            )
            previous = values[degree - 1, order]
            before = values[degree - 2, order]
            values[degree, order] = first * (points * previous - second * before)


def normalised_legendre(degrees, points):
    sines = np.sqrt(np.clip(1.0 - points**2, 0.0, None))
    values = np.zeros((degrees, degrees, len(points)))
    values[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
    diagonal_terms(values, sines)
    first_off_diagonal_terms(values, points)
    recurrence_terms(values, points)
    return values
