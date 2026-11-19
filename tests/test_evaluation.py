import itertools

import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.data.batches import IndexedDataset, evenly_spaced_indices, random_batches
from spheridiff.data.postprocess import PhysicalConverter
from spheridiff.data.statistics import Statistics
from spheridiff.data.transforms import apply_precipitation_transform
from spheridiff.diffusion.operators import SparseMask, SpatialAverage, TemporalAverage, random_sensor_mask
from spheridiff.diffusion.process import build_process
from spheridiff.diffusion.sampler import ModelDenoiser, ddim_sample
from spheridiff.evaluation.metrics import (
    RunningMean,
    crps_unbiased,
    ensemble_variance,
    mean_square_error,
)
from spheridiff.evaluation.rollout import rollout


def brute_force_crps(ensemble, truth):
    members = ensemble.shape[1]
    skill = (ensemble - truth[:, None]).abs().mean(dim=1)
    pairs = torch.zeros_like(skill)
    for first, second in itertools.product(range(members), repeat=2):
        pairs += (ensemble[:, first] - ensemble[:, second]).abs()
    return skill - pairs / (2.0 * members * (members - 1))


def test_crps_matches_the_pairwise_definition():
    generator = torch.Generator().manual_seed(0)
    ensemble = torch.randn(2, 4, 3, 2, 5, 6, generator=generator)
    truth = torch.randn(2, 3, 2, 5, 6, generator=generator)
    area = torch.ones(5, 6)
    expected = brute_force_crps(ensemble, truth).mean(dim=(-2, -1))
    assert torch.allclose(crps_unbiased(ensemble, truth, area), expected, atol=1e-5)


def test_crps_of_identical_members_is_the_mean_absolute_error():
    truth = torch.randn(2, 3, 2, 5, 6)
    ensemble = truth[:, None] + 0.5
    ensemble = ensemble.expand(-1, 3, -1, -1, -1, -1)
    area = torch.ones(5, 6)
    assert torch.allclose(crps_unbiased(ensemble, truth, area), torch.full((2, 3, 2), 0.5), atol=1e-5)


def test_a_constant_error_ignores_the_area_weights():
    grid = make_grid(8, 16)
    area = torch.as_tensor(grid.area_weights)
    prediction = torch.full((2, 3, 4, 8, 16), 2.0)
    truth = torch.zeros_like(prediction)
    assert torch.allclose(mean_square_error(prediction, truth, area), torch.full((2, 3, 4), 4.0), atol=1e-5)


def test_ensemble_variance_is_unbiased():
    ensemble = torch.randn(2, 5, 3, 2, 4, 4)
    area = torch.ones(4, 4)
    expected = ensemble.var(dim=1, unbiased=True).mean(dim=(-2, -1))
    assert torch.allclose(ensemble_variance(ensemble, area), expected, atol=1e-6)


def test_spread_skill_ratio_is_one_for_a_calibrated_ensemble():
    generator = torch.Generator().manual_seed(1)
    members = 5
    truth = torch.randn(400, 1, 1, 8, 8, generator=generator)
    ensemble = torch.randn(400, members, 1, 1, 8, 8, generator=generator)
    area = torch.ones(8, 8)
    spread = ensemble_variance(ensemble, area).mean().sqrt()
    error = mean_square_error(ensemble.mean(dim=1), truth, area).mean().sqrt()
    ratio = ((members + 1.0) / members) ** 0.5 * spread / error
    assert ratio.item() == pytest.approx(1.0, abs=0.05)


def test_running_mean_averages_over_batches():
    running = RunningMean()
    running.add(torch.ones(3, 2))
    running.add(torch.full((1, 2), 5.0))
    assert torch.allclose(running.mean(), torch.full((2,), 2.0))


def test_rollout_chains_windows_through_the_context_frame():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    first = torch.randn(2, 1, 3, 8, 16)
    auxiliary = torch.zeros(2, 9, 2, 8, 16)
    copy_first = lambda noisy, aux, level: noisy[:, :1].expand_as(noisy)
    output = rollout(copy_first, process, first, auxiliary, steps=4)
    assert output.shape == (2, 9, 3, 8, 16)
    assert torch.equal(output[:, :1], first)
    assert torch.allclose(output, first.expand(-1, 9, -1, -1, -1), atol=0.05)


def test_rollout_rejects_incomplete_windows():
    process = build_process("gaussian", "identity", make_grid(8, 16))
    with pytest.raises(ValueError):
        rollout(lambda n, a, level: n, process, torch.zeros(1, 1, 3, 8, 16), torch.zeros(1, 7, 2, 8, 16), 2)


def test_spectral_process_recovers_a_band_limited_truth():
    grid = make_grid(16, 32)
    process = build_process("spectral", "band_limit", grid)
    truth = process.projection(torch.randn(2, 5, 3, 16, 32))
    oracle = lambda noisy, aux, level: truth
    output = ddim_sample(oracle, process, torch.zeros(2, 5, 2, 16, 32), 3, 15)
    assert torch.allclose(output, truth, atol=0.02)


def test_operator_baselines_reproduce_their_observation():
    field = torch.randn(2, 5, 3, 8, 16)
    for operator in (
        SpatialAverage(4),
        TemporalAverage(5),
        SparseMask(random_sensor_mask(8, 16, 0.2, "cpu", 1)),
    ):
        observation = operator.apply(field)
        assert torch.allclose(operator.apply(operator.baseline(observation)), observation, atol=1e-5)


def test_converter_restores_physical_values():
    names = ["500_temperature", "total_precipitation_6hr"]
    statistics = Statistics(names, np.array([250.0, 0.2]), np.array([10.0, 0.1]), np.array([1.0, 0.05]))
    raw = np.stack([np.full((4, 2, 3), 260.0), np.full((4, 2, 3), 0.003)], axis=1).astype(np.float32)
    transformed = apply_precipitation_transform(raw.copy(), names, ("total_precipitation_6hr",))
    normalised = statistics.normalise(transformed.copy())
    converter = PhysicalConverter(statistics, names, ("total_precipitation_6hr",))
    restored = converter(torch.from_numpy(normalised))
    assert torch.allclose(restored, torch.from_numpy(raw), atol=1e-4)


def test_random_batches_can_start_in_the_middle():
    dataset = [{"frames": torch.tensor([float(index)])} for index in range(50)]
    full = list(itertools.islice(random_batches(dataset, 4, seed=3), 6))
    resumed = list(itertools.islice(random_batches(dataset, 4, seed=3, start_step=4), 2))
    assert torch.equal(full[4]["frames"], resumed[0]["frames"])
    assert torch.equal(full[5]["frames"], resumed[1]["frames"])


def test_indexed_dataset_and_spacing():
    base = list(range(100, 120))
    subset = IndexedDataset(base, [3, 5, 7])
    assert len(subset) == 3 and subset[1] == 105
    indices = evenly_spaced_indices(100, 5)
    assert indices[0] == 0 and indices[-1] == 99 and len(indices) == 5
    assert len(evenly_spaced_indices(3, 10)) == 3
