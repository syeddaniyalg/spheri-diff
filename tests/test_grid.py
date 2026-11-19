import numpy as np

from spheridiff.grid.grid import GridSpec


def make_grid(height, width):
    latitudes = np.linspace(-90.0, 90.0, height)
    longitudes = np.arange(width) * 360.0 / width
    return GridSpec(latitudes, longitudes)


def test_latitude_areas_sum_to_two():
    grid = make_grid(32, 64)
    assert np.isclose(grid.latitude_areas.sum(), 2.0)


def test_area_weights_have_mean_one():
    grid = make_grid(32, 64)
    assert grid.area_weights.shape == (32, 64)
    assert np.isclose(grid.area_weights.mean(), 1.0, atol=1e-6)


def test_poles_have_smallest_weight():
    weights = make_grid(32, 64).area_weights[:, 0]
    assert weights[0] == weights.min()
    assert weights[0] == weights[-1]


def test_bandlimit_of_pilot_grid():
    grid = make_grid(32, 64)
    assert grid.bandlimit == 16
    assert grid.coefficients == 256


def test_bandlimit_of_wind_grid():
    grid = make_grid(121, 240)
    assert grid.bandlimit == 61
    assert grid.coefficients == 3721
