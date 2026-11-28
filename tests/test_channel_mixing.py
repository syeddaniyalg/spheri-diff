import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.finetune.transfer import apply_adaptation, transfer_weights
from spheridiff.grid.grid import GridSpec
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spec import load_model_spec
from spheridiff.models.spheridiff.spectral import (
    DegreeChannelMixing,
    SlepianDegreeChannelMixing,
    hat_basis,
    knot_positions,
    regional_basis,
    slepian_degree_operators,
)
from spheridiff.sphere.harmonics import real_harmonics
from spheridiff.sphere.slepian import KernelSlepianTransform
from spheridiff.sphere.transform import SphericalTransform

SMALL_PATCH = GridSpec(10.0 + 3.0 * np.arange(8), 40.0 + 3.0 * np.arange(16))
SMALL_CHANNEL = {"base_width": 8, "levels": 3, "heads": 4, "spectral_precision": "fp32", "channel_knots": 4, "noise_conditioning": True, "spherical_convolution": True}
SMALL_GLOBAL = {"bandlimit": 7, "analysis": "least_squares"}
SMALL_REGIONAL = {"kind": "slepian", "method": "kernel", "bandlimit": 24}


def global_stage(degrees=7, width=3, knots=4):
    transform = SphericalTransform(make_grid(degrees + 1, 2 * degrees), degrees, "least_squares")
    return DegreeChannelMixing(transform, width, knots)


def test_the_knot_basis_sums_to_one_on_every_degree():
    basis = hat_basis(knot_positions(31, 8), 31)
    assert basis.shape == (31, 8)
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert np.allclose(hat_basis(knot_positions(7, 8), 7), np.eye(7))


def test_the_regional_basis_keeps_pretrained_degrees_and_truncates_above_them():
    basis, pretrained = regional_basis(120, 8, 31)
    assert pretrained == 8 and basis.shape == (120, 16)
    assert np.allclose(basis[:31, :8], hat_basis(knot_positions(31, 8), 31))
    assert np.all(basis[31:, :8] == 0.0)
    assert np.allclose(basis[:31, 8:], 0.0)
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert np.isclose(basis[31, 8], 1.0)


def test_identity_weights_give_the_identity_operator():
    stage = global_stage()
    stage.isotropic.data[:] = torch.eye(3)
    coefficients = torch.randn(2, 49, 3)
    assert torch.allclose(stage.mix(coefficients), coefficients, atol=1e-6)


def test_the_zonal_part_is_the_longitude_derivative():
    degrees, width = 7, 1
    stage = global_stage(degrees, width)
    stage.zonal.data[:] = 1.0
    torch.manual_seed(0)
    coefficients = torch.randn(degrees * degrees, width, dtype=torch.float64)
    latitudes, longitudes, step = np.array([-40.0, 0.0, 25.0]), np.array([10.0, 100.0, 250.0]), 1e-4
    field = lambda shift: real_harmonics(latitudes, longitudes + shift, degrees) @ coefficients.numpy()
    derivative = (field(step) - field(-step)) / (2.0 * np.radians(step))
    mixed = stage.mix(coefficients.float()).detach().double().numpy()
    assert np.allclose(real_harmonics(latitudes, longitudes, degrees) @ mixed, derivative / degrees, atol=1e-4)


def test_the_isotropic_part_commutes_with_rotation_about_the_pole():
    stage = global_stage()
    torch.manual_seed(1)
    stage.isotropic.data.normal_()
    stage.zonal.data.normal_()
    latitudes, longitudes = np.linspace(-80.0, 80.0, 9), np.arange(16) * 22.5
    synthesis = torch.tensor(real_harmonics(latitudes, longitudes, 7), dtype=torch.float32)
    analysis = torch.linalg.pinv(synthesis)
    field = torch.randn(49, 3)
    shifted = torch.tensor(real_harmonics(latitudes, longitudes + 45.0, 7), dtype=torch.float32)
    rotated = analysis @ (shifted @ field)
    assert torch.allclose(stage.mix(rotated), analysis @ (shifted @ stage.mix(field)), atol=1e-3)


def test_the_regional_operators_match_a_dense_computation():
    transform = KernelSlepianTransform(SMALL_PATCH, 20)
    basis, _ = regional_basis(20, 4, 8)
    operators = slepian_degree_operators(transform, basis, 8)
    harmonic = transform.harmonic_basis(400).double().numpy()
    degrees = np.floor(np.sqrt(np.arange(400))).astype(int)
    orders = np.concatenate([np.arange(-degree, degree + 1) for degree in range(20)])
    partner = degrees * degrees + degrees - orders
    derivative = np.zeros((400, 400))
    derivative[np.arange(400), partner] = orders / 8.0
    count = basis.shape[1]
    for knot in range(count):
        scaled = np.diag(basis[degrees, knot])
        assert np.allclose(operators[knot], harmonic.T @ scaled @ harmonic, atol=1e-8)
        assert np.allclose(operators[count + knot], harmonic.T @ scaled @ derivative @ harmonic, atol=1e-8)


def test_regional_identity_weights_give_the_identity_on_slepian_coefficients():
    transform = KernelSlepianTransform(SMALL_PATCH, 20)
    stage = SlepianDegreeChannelMixing(transform, 2, 4, 8)
    stage.isotropic.data[:] = torch.eye(2)
    stage.isotropic_new.data[:] = torch.eye(2)
    coefficients = torch.randn(transform.coefficients, 2)
    assert torch.allclose(stage.mix(coefficients), coefficients, atol=1e-4)


def test_the_planned_model_has_no_mode_mixing_weight():
    spec = load_model_spec("spheridiff_pooled3_channel")
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == 2415897
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.endswith((".weight", ".kappa", ".diagonal")) and ".spectral." in name for name in names)
    regional = load_model_spec("spheridiff_pooled3_channel_regional")
    assert regional.architecture["pretrained_degrees"] == [31, 15, 7] and regional.sphere["bandlimit"] == 120


def test_channel_mixing_excludes_mode_mixing_options():
    with pytest.raises(ValueError):
        build_model("spheridiff_pooled", {**SMALL_CHANNEL, "coefficient_filters": True}, ModelContext(6, 3, 5, make_grid(8, 16), SMALL_GLOBAL))


def test_transfer_freezes_pretrained_knots_and_starts_new_knots_and_delta_at_zero():
    torch.manual_seed(2)
    pretrained = build_model("spheridiff_pooled", SMALL_CHANNEL, ModelContext(6, 3, 5, make_grid(8, 16), SMALL_GLOBAL))
    for parameter in pretrained.parameters():
        parameter.data.normal_(0.0, 0.05)
    architecture = {**SMALL_CHANNEL, "pretrained_degrees": [7, 3, 1]}
    regional = build_model("spheridiff_pooled", architecture, ModelContext(6, 3, 5, SMALL_PATCH, SMALL_REGIONAL))
    transfer_weights(regional, pretrained.state_dict())
    apply_adaptation(regional, "delta_and_pointwise")
    stage = regional.down[0].spectral
    assert torch.equal(stage.isotropic, pretrained.down[0].spectral.isotropic)
    assert not stage.isotropic.requires_grad and not stage.zonal.requires_grad
    assert stage.isotropic_new.requires_grad and torch.count_nonzero(stage.isotropic_new) == 0
    assert torch.count_nonzero(stage.delta) == 0 and stage.delta.requires_grad
    assert regional.down[0].local.convolution.weight.requires_grad
    noisy, auxiliary = torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16)
    regional(noisy, auxiliary).square().mean().backward()
    assert stage.isotropic.grad is None and stage.isotropic_new.grad is not None
