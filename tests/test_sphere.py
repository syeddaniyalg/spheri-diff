import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.sphere.harmonics import coefficient_index, real_harmonics
from spheridiff.sphere.quadrature import clenshaw_curtis_weights, sphere_quadrature
from spheridiff.sphere.transform import SphericalTransform


@pytest.fixture(scope="module")
def transform():
    return SphericalTransform(make_grid(32, 64))


def test_quadrature_weights_sum_to_two():
    assert np.isclose(clenshaw_curtis_weights(32).sum(), 2.0)


def test_quadrature_is_exact_for_polynomials():
    points = 32
    nodes = np.cos(np.pi * np.arange(points) / (points - 1))
    weights = clenshaw_curtis_weights(points)
    for power in range(points):
        exact = 0.0 if power % 2 else 2.0 / (power + 1)
        assert np.isclose(weights @ nodes**power, exact, atol=1e-10)


def test_sphere_quadrature_covers_the_sphere():
    assert np.isclose(sphere_quadrature(make_grid()).sum(), 4.0 * np.pi)


def test_low_degree_harmonics_match_closed_forms():
    grid = make_grid()
    matrix = real_harmonics(grid.latitudes, grid.longitudes, 2)
    colatitude = np.radians(90.0 - grid.latitudes)[:, None]
    longitude = np.radians(grid.longitudes)[None, :]
    expected = {
        coefficient_index(0, 0): np.full((32, 64), 1.0 / np.sqrt(4.0 * np.pi)),
        coefficient_index(1, 0): np.sqrt(3.0 / (4.0 * np.pi)) * np.cos(colatitude) + 0 * longitude,
        coefficient_index(1, 1): np.sqrt(3.0 / (4.0 * np.pi)) * np.sin(colatitude) * np.cos(longitude),
        coefficient_index(1, -1): np.sqrt(3.0 / (4.0 * np.pi)) * np.sin(colatitude) * np.sin(longitude),
    }
    for index, values in expected.items():
        assert np.allclose(matrix[:, index].reshape(32, 64), values, atol=1e-10)


def test_analysis_inverts_synthesis(transform):
    product = transform.synthesis @ transform.analysis
    identity = torch.eye(transform.coefficients)
    assert torch.allclose(product, identity, atol=1e-5)


def test_band_limit_is_a_lossy_projection(transform):
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(3, 32, 64, generator=generator)
    once = transform.band_limit(field)
    twice = transform.band_limit(once)
    assert torch.allclose(once, twice, atol=1e-4)
    assert (field - once).abs().max() > 0.1


def test_harmonic_field_is_recovered_exactly(transform):
    coefficients = torch.zeros(transform.coefficients)
    coefficients[coefficient_index(5, -3)] = 2.0
    field = transform.synthesise(coefficients)
    assert torch.allclose(transform.analyse(field), coefficients, atol=1e-5)


def test_squared_harmonics_sum_to_a_constant(transform):
    total = (transform.synthesis**2).sum(dim=0)
    expected = transform.coefficients / (4.0 * np.pi)
    assert torch.allclose(total, torch.full_like(total, expected), atol=1e-4)


def test_shapes_follow_leading_dimensions(transform):
    field = torch.randn(2, 5, 7, 32, 64)
    coefficients = transform.analyse(field)
    assert coefficients.shape == (2, 5, 7, 256)
    assert transform.synthesise(coefficients).shape == field.shape
