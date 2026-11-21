import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.diffusion.process import build_process
from spheridiff.experiments.spec import load_experiment
from spheridiff.grid.grid import gauss_legendre_grid
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spec import load_model_spec
from spheridiff.models.spheridiff.config import SpherIDiffConfig
from spheridiff.models.spheridiff.spectral import DiagonalLowRankMixing, SpectralMixing
from spheridiff.sphere.quadrature import sphere_quadrature
from spheridiff.sphere.separable import SeparableTransform
from spheridiff.sphere.transform import SphericalTransform, build_transform
from spheridiff.training.settings import load_train_settings

SIZES = {"xs": 264987, "s": 335901, "m": 587805, "b4": 1589277}
SMALL_SPHERE = {"bandlimit": 12, "analysis": "least_squares"}
SMALL_ARCHITECTURE = {"widths": [16, 32, 16], "heads": 4, "rank": 4, "internal_grid": "gauss", "transform": "separable"}


def small_context(channels=6, auxiliary=3):
    return ModelContext(channels, auxiliary, 5, make_grid(16, 32), SMALL_SPHERE)


def test_gauss_grid_supports_the_full_latitude_band_limit():
    grid = gauss_legendre_grid(32, 64)
    assert grid.bandlimit == 32
    assert np.isclose(sphere_quadrature(grid).sum(), 4.0 * np.pi)
    transform = SphericalTransform(grid, 31)
    product = transform.analysis.double().T @ transform.synthesis.double().T
    assert torch.allclose(product, torch.eye(961, dtype=torch.float64), atol=1e-5)


def test_least_squares_analysis_inverts_the_synthesis_at_band_limit_31():
    transform = SphericalTransform(make_grid(32, 64), 31, "least_squares")
    coefficients = torch.randn(3, 961)
    assert torch.allclose(transform.analyse(transform.synthesise(coefficients)), coefficients, atol=1e-4)


def test_least_squares_agrees_with_quadrature_where_quadrature_is_exact():
    grid = make_grid(32, 64)
    quadrature, least_squares = SphericalTransform(grid), SphericalTransform(grid, 16, "least_squares")
    assert torch.allclose(quadrature.analysis, least_squares.analysis, atol=1e-6)


def test_quadrature_rejects_degrees_beyond_its_limit():
    with pytest.raises(ValueError):
        SphericalTransform(make_grid(32, 64), 31)
    with pytest.raises(ValueError):
        SphericalTransform(make_grid(32, 64), 31, "unknown")


def test_separable_transform_matches_the_dense_transform():
    grid = gauss_legendre_grid(32, 64)
    dense, separable = SphericalTransform(grid, 31), SeparableTransform(grid, 31)
    values = torch.randn(2, 3, 2048, 4)
    coefficients = torch.randn(2, 3, 961, 4)
    assert torch.allclose(separable.analyse_points(values), dense.analyse_points(values), atol=1e-5)
    assert torch.allclose(separable.synthesise_points(coefficients), dense.synthesise_points(coefficients), atol=1e-4)


def test_separable_transform_needs_longitudes_from_zero():
    grid = gauss_legendre_grid(16, 32)
    shifted = type(grid)(grid.latitudes, grid.longitudes + 1.0, grid.quadrature)
    with pytest.raises(ValueError):
        SeparableTransform(shifted, 12)


def test_structured_weight_acts_as_its_dense_matrix():
    torch.manual_seed(0)
    transform = SeparableTransform(gauss_legendre_grid(16, 32), 12)
    structured = DiagonalLowRankMixing(transform, 4)
    for parameter in structured.parameters():
        parameter.data.normal_()
    dense = SpectralMixing(transform)
    dense.weight.data.copy_(structured.dense_weight())
    values = torch.randn(2, 5, 512, 3)
    assert torch.allclose(structured(values), dense(values), atol=1e-4)
    assert sum(parameter.numel() for parameter in structured.parameters()) == 144 + 2 * 144 * 4


def test_resampling_keeps_band_limited_fields():
    model = build_model("spheridiff", SMALL_ARCHITECTURE, small_context())
    data_transform = build_transform(make_grid(16, 32), SMALL_SPHERE)
    field = data_transform.synthesise_points(torch.randn(2, 144, 3))
    moved = model.grids.leave(model.grids.enter(field))
    assert torch.allclose(moved, field, atol=1e-4)


def test_amended_model_starts_at_zero_and_every_parameter_learns():
    torch.manual_seed(0)
    model = build_model("spheridiff", SMALL_ARCHITECTURE, small_context())
    noisy, auxiliary = torch.randn(2, 5, 6, 16, 32), torch.randn(2, 5, 3, 16, 32)
    assert torch.count_nonzero(model(noisy, auxiliary)) == 0
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    target = torch.randn(2, 5, 6, 16, 32)
    for _ in range(10):
        optimizer.zero_grad()
        ((model(noisy, auxiliary) - target) ** 2).mean().backward()
        optimizer.step()
    unchanged = [name for (name, parameter), start in zip(model.named_parameters(), initial) if torch.equal(parameter, start)]
    assert unchanged == []


def test_amended_options_are_checked():
    with pytest.raises(ValueError):
        SpherIDiffConfig((16, 32), transform="separable")
    with pytest.raises(ValueError):
        SpherIDiffConfig((16, 32), internal_grid="healpix")
    with pytest.raises(ValueError):
        SpherIDiffConfig((16, 32), rank=0)


def test_spectral_noise_follows_the_sphere_settings():
    process = build_process("spectral", "band_limit", make_grid(32, 64), {"bandlimit": 31, "analysis": "least_squares"})
    assert process.noise.transform.coefficients == 961
    noise = process.noise.sample((2, 3, 32, 64), torch.device("cpu"))
    assert torch.allclose(process.projection(noise), noise, atol=1e-4)


@pytest.mark.parametrize("size", sorted(SIZES))
def test_band_limit_31_configs_have_the_planned_sizes(size):
    spec = load_model_spec(f"spheridiff_l31_{size}")
    assert spec.sphere == {"bandlimit": 31, "analysis": "least_squares"}
    context = ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere)
    model = build_model(spec.name, spec.architecture, context)
    assert count_parameters(model) == SIZES[size]


def test_rank_128_ablation_has_the_planned_size():
    spec = load_model_spec("spheridiff_l31_b4_r128")
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == 2327325


def test_earlier_configs_keep_the_dense_model():
    spec = load_model_spec("spheridiff_b4")
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == 1601561
    assert model.grids.incoming is None and isinstance(model.blocks[0].spectral, SpectralMixing)


def test_band_limit_31_experiments_change_only_the_length_of_training():
    pilot, short = load_train_settings("pilot"), load_train_settings("pilot_25k")
    assert (short.steps, short.evaluate_every, short.checkpoint_every) == (25000, 1250, 1250)
    for name in ("batch_size", "learning_rate", "min_learning_rate", "warmup_fraction", "weight_decay"):
        assert getattr(short, name) == getattr(pilot, name)
    for name in ("gradient_clip", "ema_decay", "precision", "milestones", "evaluation", "source_levels"):
        assert getattr(short, name) == getattr(pilot, name)
    for size in SIZES:
        spec = load_experiment(f"pilot_spheridiff_l31_{size}")
        assert (spec.model, spec.train, spec.evaluation) == (f"spheridiff_l31_{size}", "pilot_25k", "pilot")
