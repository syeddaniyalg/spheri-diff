import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.data.statistics import Statistics
from spheridiff.diffusion.guidance import MMPSDenoiser
from spheridiff.diffusion.solvers import conjugate_gradient
from spheridiff.diffusion.objective import denoising_loss, fixed_level_loss
from spheridiff.diffusion.operators import (
    SparseMask,
    SpatialAverage,
    TemporalAverage,
    random_sensor_mask,
)
from spheridiff.diffusion.process import build_process
from spheridiff.diffusion.sampler import ModelDenoiser, ddim_sample
from spheridiff.diffusion.schedule import RectifiedSchedule
from spheridiff.diffusion.weights import channel_weights, loss_weights, pressure_level

SOURCE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


class OracleDenoiser:
    def __init__(self, truth):
        self.truth = truth

    def __call__(self, noisy, auxiliary, level):
        return self.truth


def test_schedule_endpoints():
    schedule = RectifiedSchedule()
    assert schedule(0.0) == pytest.approx((1.0, 0.001))
    assert schedule(1.0) == pytest.approx((0.001, 1.0))
    alpha, sigma = schedule(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(alpha, torch.tensor([1.0, 0.5005, 0.001]))
    assert torch.allclose(sigma, torch.tensor([0.001, 0.5005, 1.0]))


def test_corruption_follows_the_levels_per_frame():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    clean = torch.ones(2, 3, 4, 8, 16)
    levels = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    noisy, target = process.corrupt(clean, levels, torch.Generator().manual_seed(0))
    assert torch.equal(target, clean)
    assert torch.allclose(noisy[0, 0], clean[0, 0], atol=0.01)
    assert noisy[0, 2].std() > 0.9
    assert noisy[1, 1].sub(1.0).abs().max() < 0.01
    assert not torch.allclose(noisy[0, 1] - 0.5, noisy[1, 2] - 0.5)


def test_spectral_process_projects_the_target_and_keeps_unit_noise():
    grid = make_grid(32, 64)
    process = build_process("spectral", "band_limit", grid)
    clean = torch.randn(2, 3, 4, 32, 64)
    levels = torch.ones(2, 3)
    noisy, target = process.corrupt(clean, levels, torch.Generator().manual_seed(1))
    assert not torch.allclose(target, clean, atol=0.1)
    assert torch.allclose(process.projection(target), target, atol=1e-4)
    assert noisy.std().item() == pytest.approx(1.0, abs=0.05)
    again, _ = process.corrupt(clean, levels, torch.Generator().manual_seed(1))
    assert torch.equal(noisy, again)


def test_channel_weights_follow_pressure_levels():
    names = ["500_temperature", "850_temperature", "2m_temperature", "mean_sea_level_pressure"]
    values = channel_weights(names, SOURCE_LEVELS)
    assert values[0].item() == pytest.approx(500 / sum(SOURCE_LEVELS))
    assert values[1].item() == pytest.approx(850 / sum(SOURCE_LEVELS))
    assert values[2].item() == 1.0
    assert values[3].item() == pytest.approx(0.1)
    assert pressure_level("2m_temperature") is None


def test_loss_weights_combine_area_and_statistics():
    grid = make_grid(8, 16)
    names = ["500_temperature", "total_precipitation_6hr"]
    statistics = Statistics(
        names, np.zeros(2), np.array([2.0, 1.0]), np.array([1.0, 1.0])
    )
    weights = loss_weights(grid, names, statistics, SOURCE_LEVELS)
    assert weights.shape == (2, 8, 16)
    ratio = weights[0] / torch.as_tensor(grid.area_weights)
    assert ratio.std() < 1e-6


def test_zero_model_loss_equals_weighted_target_energy():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    clean = torch.randn(2, 3, 4, 8, 16)
    weights = torch.rand(4, 8, 16) + 0.5
    zero_model = lambda noisy, auxiliary: torch.zeros_like(noisy)
    loss = denoising_loss(zero_model, process, weights, clean, None)
    assert loss.item() == pytest.approx((clean**2 * weights).mean().item())


def test_fixed_level_loss_is_reproducible():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    clean = torch.randn(2, 3, 4, 8, 16)
    weights = torch.ones(4, 8, 16)
    model = lambda noisy, auxiliary: 0.5 * noisy
    first = fixed_level_loss(model, process, weights, clean, None, 0.5, torch.Generator().manual_seed(3))
    second = fixed_level_loss(model, process, weights, clean, None, 0.5, torch.Generator().manual_seed(3))
    assert first.item() == second.item()


def test_ddim_recovers_the_truth_with_a_perfect_denoiser():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    truth = torch.randn(2, 5, 4, 8, 16)
    auxiliary = torch.zeros(2, 5, 3, 8, 16)
    output = ddim_sample(OracleDenoiser(truth), process, auxiliary, 4, 15)
    assert torch.allclose(output, truth, atol=0.02)


def test_ddim_keeps_the_context_frames():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    truth = torch.randn(2, 5, 4, 8, 16)
    mask = torch.zeros_like(truth, dtype=torch.bool)
    mask[:, :1] = True
    context = torch.where(mask, truth, torch.zeros_like(truth))
    auxiliary = torch.zeros(2, 5, 3, 8, 16)
    output = ddim_sample(ModelDenoiser(lambda x, a: x), process, auxiliary, 4, 5, context, mask)
    assert torch.equal(output[:, :1], truth[:, :1])


def test_ddim_is_reproducible_with_a_generator():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    auxiliary = torch.zeros(1, 5, 3, 8, 16)
    denoiser = ModelDenoiser(lambda x, a: 0.5 * x)
    first = ddim_sample(denoiser, process, auxiliary, 4, 4, generator=torch.Generator().manual_seed(5))
    second = ddim_sample(denoiser, process, auxiliary, 4, 4, generator=torch.Generator().manual_seed(5))
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "operator",
    [SpatialAverage(4), TemporalAverage(5), SparseMask(random_sensor_mask(8, 16, 0.1, "cpu", 0))],
)
def test_operators_satisfy_the_adjoint_identity(operator):
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(2, 5, 3, 8, 16, generator=generator)
    observation = torch.randn_like(operator.apply(field))
    left = (operator.apply(field) * observation).sum()
    right = (field * operator.adjoint(observation)).sum()
    assert left.item() == pytest.approx(right.item(), rel=1e-5)


def test_operator_output_shapes():
    field = torch.randn(2, 5, 3, 8, 16)
    assert SpatialAverage(4).apply(field).shape == (2, 5, 3, 2, 4)
    assert TemporalAverage(5).apply(field).shape == (2, 1, 3, 8, 16)


def test_sensor_mask_has_the_requested_count():
    mask = random_sensor_mask(32, 64, 0.05, "cpu", 0)
    assert mask.sum().item() == int(np.ceil(0.05 * 32 * 64))
    assert torch.equal(mask, random_sensor_mask(32, 64, 0.05, "cpu", 0))


def test_conjugate_gradient_solves_a_small_system():
    generator = torch.Generator().manual_seed(0)
    matrix = torch.randn(6, 6, generator=generator)
    matrix = matrix @ matrix.T + torch.eye(6)
    rhs = torch.randn(2, 6, generator=generator)
    solution = conjugate_gradient(lambda v: v @ matrix, rhs, 6)
    assert torch.allclose(solution @ matrix, rhs, atol=1e-3)


def test_mmps_matches_the_closed_form_for_an_identity_denoiser():
    schedule = RectifiedSchedule()
    operator = SparseMask(torch.ones(1, 1, 1, 8, 16))
    noisy = torch.randn(2, 5, 3, 8, 16)
    observation = torch.randn(2, 5, 3, 8, 16)
    denoiser = MMPSDenoiser(ModelDenoiser(lambda x, a: x), schedule, operator, observation, 0.0015, 2)
    level = 0.5
    alpha, sigma = schedule(level)
    gamma = sigma**2 / alpha
    expected = noisy + gamma / (0.0015 + gamma) * (observation - noisy)
    assert torch.allclose(denoiser(noisy, None, level), expected, atol=1e-4)


def test_mmps_pulls_the_estimate_towards_the_observation():
    schedule = RectifiedSchedule()
    operator = TemporalAverage(5)
    noisy = torch.randn(1, 5, 3, 8, 16)
    observation = torch.full((1, 1, 3, 8, 16), 2.0)
    base = ModelDenoiser(lambda x, a: 0.5 * x)
    guided = MMPSDenoiser(base, schedule, operator, observation, 0.0015, 2)(noisy, None, 0.5)
    plain = base(noisy, None, 0.5)
    before = (operator.apply(plain) - observation).abs().mean()
    after = (operator.apply(guided) - observation).abs().mean()
    assert after < before
