import copy
import json
import math

import numpy as np
import pytest
import torch
import yaml

from helpers import make_grid
from spheridiff.common.config import load_data_config
from pipeline import EVALUATION_FIELDS, TINY_SPHERIDIFF, TINY_WIND, dump, write_experiment
from spheridiff.analysis.final_table import build_rows, format_table, write_final_summary
from spheridiff.diffusion.operators import (
    FrameMask,
    SparseMask,
    SpatioTemporalAverage,
    interpolation_weights,
    random_space_time_mask,
)
from spheridiff.diffusion.guidance import MMPSDenoiser
from spheridiff.diffusion.process import build_process
from spheridiff.diffusion.solvers import solve_linear
from spheridiff.diffusion.sampler import ModelDenoiser, ddim_sample, native_steps, noised_guess
from spheridiff.evaluation.protocol import load_protocol, protocol_digest, protocol_from_dict
from spheridiff.evaluation.reconstruction import unobserved_error
from spheridiff.experiments.build import build_trainer
from spheridiff.experiments.final import RESULT_NAME, final_datasets, run_final_evaluation
from spheridiff.experiments.spec import load_experiment
from spheridiff.kaggle.jobs import final_job

CPU = torch.device("cpu")
MODELS = {"wind": TINY_WIND, "spheridiff": TINY_SPHERIDIFF}
TASKS = (
    "spatial_reconstruction",
    "spatial_downsampling",
    "temporal_reconstruction",
    "temporal_downsampling",
    "spatiotemporal_downsampling",
    "spatiotemporal_reconstruction",
)


def tiny_protocol_dict(name="tiny_protocol"):
    return {
        "name": name,
        "split": "val",
        "windows": 2,
        "members": 2,
        "ddim_steps": 3,
        "batch_size": 2,
        "forecast_frames": 9,
        "seed": 42,
        "precision": "fp32",
        "fields": EVALUATION_FIELDS,
        "guidance": {"noise_variance": 0.0015, "iterations": 2, "solver": "gmres"},
        "native_init": {"start_level": 0.5},
        "methods": ["mmps", "native", "native_init"],
        "tasks": {
            "spatial_reconstruction": {"operator": "sensor_mask", "fraction": 0.1},
            "spatial_downsampling": {"operator": "spatial_average", "factor": 4},
            "temporal_reconstruction": {"operator": "frame_mask", "observed": [0, 4]},
            "temporal_downsampling": {"operator": "temporal_average"},
            "spatiotemporal_downsampling": {"operator": "spatiotemporal_average", "factor": 4},
            "spatiotemporal_reconstruction": {"operator": "space_time_mask", "fraction": 0.1},
        },
    }


def trained_run(tmp_path, kind):
    experiment, cache_root = write_experiment(tmp_path, MODELS[kind], steps=2)
    trainer, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    protocol = load_protocol(dump(tmp_path / "protocol.yaml", tiny_protocol_dict()))
    return experiment, cache_root, protocol


def test_the_final_protocol_defines_the_five_tasks_and_both_methods():
    protocol = load_protocol("final_v1")
    assert tuple(protocol.tasks) == TASKS
    assert protocol.methods == ("mmps", "native", "native_init")
    assert protocol.split == "test" and protocol.seed == 42
    assert protocol.members >= 2 and (protocol.forecast_frames - 1) % 4 == 0


def test_the_protocol_digest_is_stable_and_changes_with_any_setting():
    raw = tiny_protocol_dict()
    assert protocol_from_dict(raw).digest == protocol_from_dict(copy.deepcopy(raw)).digest
    changed = copy.deepcopy(raw)
    changed["native_init"]["start_level"] = 0.6
    assert protocol_from_dict(changed).digest != protocol_from_dict(raw).digest
    assert protocol_digest(raw) == protocol_from_dict(raw).digest


def test_invalid_protocols_are_rejected():
    raw = tiny_protocol_dict()
    raw["tasks"]["bad"] = {"operator": "unknown"}
    with pytest.raises(ValueError):
        protocol_from_dict(raw)
    raw = tiny_protocol_dict()
    raw["members"] = 1
    with pytest.raises(ValueError):
        protocol_from_dict(raw)
    raw = tiny_protocol_dict()
    raw["methods"] = ["mmps", "other"]
    with pytest.raises(ValueError):
        protocol_from_dict(raw)


def test_frame_mask_adjoint_and_linear_interpolation_baseline():
    operator = FrameMask(5, [0, 4], "cpu")
    field = torch.randn(2, 5, 3, 4, 4)
    observation = torch.randn_like(field)
    assert (operator.apply(field) * observation).sum().item() == pytest.approx(
        (field * operator.adjoint(observation)).sum().item(), rel=1e-5
    )
    times = torch.arange(5.0).view(1, 5, 1, 1, 1)
    linear = 2.0 + 3.0 * times.expand(2, 5, 3, 4, 4)
    assert torch.allclose(operator.baseline(operator.apply(linear)), linear, atol=1e-5)
    assert interpolation_weights(5, [0, 4]).sum(dim=1).allclose(torch.ones(5))


def test_space_time_mask_has_the_requested_count_and_is_reproducible():
    mask = random_space_time_mask(5, 8, 16, 0.05, "cpu", 42)
    assert mask.shape == (1, 5, 1, 8, 16)
    assert mask.sum().item() == math.ceil(0.05 * 5 * 8 * 16)
    assert torch.equal(mask, random_space_time_mask(5, 8, 16, 0.05, "cpu", 42))
    assert not torch.equal(mask, random_space_time_mask(5, 8, 16, 0.05, "cpu", 7))


def test_unobserved_error_counts_only_the_unobserved_entries():
    area = torch.ones(4, 4)
    mask = torch.zeros(1, 1, 1, 4, 4)
    mask[..., :2, :] = 1.0
    errors = torch.zeros(2, 3, 2, 4, 4)
    errors[..., :2, :] = 100.0
    errors[..., 2:, :] = 0.5
    result = unobserved_error(errors, mask, area)
    assert torch.allclose(result, torch.full((2,), 0.5))


def test_native_denoising_starts_from_the_noised_guess_and_uses_fewer_steps():
    process = build_process("gaussian", "identity", make_grid(8, 16))
    truth = torch.randn(2, 5, 3, 8, 16)
    calls = []

    def oracle(noisy, auxiliary, level):
        calls.append(level)
        return truth

    generator = torch.Generator().manual_seed(0)
    initial = noised_guess(process, truth + 0.3, 0.5, generator)
    steps = native_steps(15, 0.5)
    output = ddim_sample(oracle, process, torch.zeros(2, 5, 2, 8, 16), 3, steps, start_level=0.5, initial=initial)
    assert steps == 8 and len(calls) == 8
    assert calls[0] == pytest.approx(0.5) and max(calls) <= 0.5 + 1e-9
    assert torch.allclose(output, truth, atol=0.02)
    assert native_steps(15, 0.01) == 2


@pytest.mark.parametrize("kind", ["wind", "spheridiff"])
def test_the_final_evaluation_covers_every_task_and_method(tmp_path, kind):
    experiment, cache_root, protocol = trained_run(tmp_path, kind)
    result = run_final_evaluation(experiment, protocol, tmp_path / "runs", cache_root, CPU)
    assert set(result["tasks"]) == set(TASKS)
    for task in TASKS:
        assert set(result["tasks"][task]) == {"baseline", "mmps", "native", "native_init"}
        for method, entry in result["tasks"][task].items():
            assert all(math.isfinite(entry[key]) for key in ("nrmse", "nrmse_unobserved", "consistency"))
            assert len(entry["window_nmse"]) == 2
            assert set(entry["rmse"]) == set(EVALUATION_FIELDS)
            assert (entry["ncrps"] is None) == (method == "baseline")
    assert result["forecast"]["ssr/500_temperature/24h"] > 0
    assert len(result["forecast"]["window_nmse"]) == 2
    assert result["protocol"]["digest"] == protocol.digest
    assert result["run"]["trained_steps"] == 2 and result["run"]["training_complete"]
    saved = json.loads((tmp_path / "runs" / "tiny" / RESULT_NAME).read_text())
    assert saved["protocol"]["name"] == "tiny_protocol"
    assert len(saved["windows"]["tasks"]["indices"]) == 2
    assert len(saved["windows"]["forecast"]["start_seconds"]) == 2


def test_the_final_evaluation_is_reproducible(tmp_path):
    experiment, cache_root, protocol = trained_run(tmp_path, "spheridiff")
    config = load_data_config(load_experiment(experiment).data, cache_root)
    datasets = final_datasets(config, protocol)
    first = run_final_evaluation(experiment, protocol, tmp_path / "runs", cache_root, CPU, datasets)
    second = run_final_evaluation(experiment, protocol, tmp_path / "runs", cache_root, CPU, datasets)
    assert first["tasks"] == second["tasks"]
    assert first["forecast"] == second["forecast"]
    assert first["windows"] == second["windows"]


def test_mask_tasks_report_errors_on_unobserved_entries_only(tmp_path):
    experiment, cache_root, protocol = trained_run(tmp_path, "wind")
    result = run_final_evaluation(experiment, protocol, tmp_path / "runs", cache_root, CPU)
    for task in ("spatial_reconstruction", "temporal_reconstruction", "spatiotemporal_reconstruction"):
        baseline = result["tasks"][task]["baseline"]
        assert baseline["nrmse_unobserved"] > baseline["nrmse"]
    for task in ("spatial_downsampling", "temporal_downsampling", "spatiotemporal_downsampling"):
        entry = result["tasks"][task]["mmps"]
        assert entry["nrmse_unobserved"] == pytest.approx(entry["nrmse"])


def test_the_summary_table_lists_runs_and_the_task_reference(tmp_path):
    experiment, cache_root, protocol = trained_run(tmp_path, "wind")
    run_final_evaluation(experiment, protocol, tmp_path / "runs", cache_root, CPU)
    text = write_final_summary(tmp_path / "runs", [(tmp_path / "runs", ["tiny"])])
    assert "tiny_protocol" in text and "task reference" in text
    assert "spatial_downsampling/mmps" in text and "temporal_reconstruction/native_init" in text
    assert (tmp_path / "runs" / "final_evaluation_summary.json").exists()
    assert write_final_summary(tmp_path / "runs", [(tmp_path / "runs", ["absent"])]) is None


def test_tables_refuse_results_from_different_protocols(tmp_path):
    experiment, cache_root, protocol = trained_run(tmp_path, "wind")
    result = run_final_evaluation(experiment, protocol, tmp_path / "runs", cache_root, CPU)
    other = copy.deepcopy(result)
    other["protocol"]["digest"] = "different"
    with pytest.raises(ValueError):
        build_rows([("one", result), ("two", other)])
    moved = copy.deepcopy(result)
    moved["windows"]["tasks"]["indices"] = [0, 1]
    with pytest.raises(ValueError):
        build_rows([("one", result), ("two", moved)])
    assert "one" in format_table(build_rows([("one", result)]))


def test_final_jobs_use_their_own_log_names():
    job = final_job("pilot_wind_s")
    assert job.name == "final_pilot_wind_s" and job.script == "final_evaluation.py"
    assert job.arguments == ("--experiments", "pilot_wind_s", "--protocol", "final_v1")
    assert not job.budgeted


def spd_system(size=7):
    generator = torch.Generator().manual_seed(1)
    matrix = torch.randn(size, size, generator=generator)
    return matrix @ matrix.T + torch.eye(size), torch.randn(3, size, generator=generator)


@pytest.mark.parametrize("solver", ["cg", "gmres"])
def test_both_solvers_recover_the_exact_solution_with_enough_iterations(solver):
    matrix, rhs = spd_system()
    solution = solve_linear(lambda v: v @ matrix, rhs, 7, solver)
    assert torch.allclose(solution @ matrix, rhs, atol=1e-3)


def test_two_iteration_solvers_reduce_the_residual_and_reject_unknown_names():
    matrix, rhs = spd_system()
    for solver in ("cg", "gmres"):
        solution = solve_linear(lambda v: v @ matrix, rhs, 2, solver)
        assert (solution @ matrix - rhs).norm() < rhs.norm()
    with pytest.raises(ValueError):
        solve_linear(lambda v: v, rhs, 2, "other")


def test_spatiotemporal_average_adjoint_and_baseline():
    operator = SpatioTemporalAverage(4, 5)
    field = torch.randn(2, 5, 3, 8, 16)
    observation = torch.randn(2, 1, 3, 2, 4)
    assert operator.apply(field).shape == observation.shape
    assert (operator.apply(field) * observation).sum().item() == pytest.approx(
        (field * operator.adjoint(observation)).sum().item(), rel=1e-5
    )
    constant = torch.full((1, 5, 2, 8, 16), 3.0)
    assert torch.allclose(operator.baseline(operator.apply(constant)), constant)


def test_the_spectral_covariance_matches_the_noise_it_describes():
    grid = make_grid(8, 16)
    process = build_process("spectral", "band_limit", grid)
    covariance = process.noise.covariance(CPU)
    first, second = torch.randn(2, 1, 2, 8, 16), torch.randn(2, 1, 2, 8, 16)
    assert (covariance.apply(first) * second).sum().item() == pytest.approx(
        (first * covariance.apply(second)).sum().item(), rel=1e-4
    )
    identity = build_process("gaussian", "identity", grid).noise.covariance(CPU)
    assert torch.equal(identity.apply(first), first)


def test_mmps_with_spectral_noise_uses_the_covariance_in_the_correction():
    grid = make_grid(8, 16)
    process = build_process("spectral", "band_limit", grid)
    covariance = process.noise.covariance(CPU)
    operator = SparseMask(torch.ones(1, 1, 1, 8, 16))
    noisy = process.projection(torch.randn(2, 5, 3, 8, 16))
    observation = process.projection(torch.randn(2, 5, 3, 8, 16))
    level = 0.5
    alpha, sigma = process.schedule(level)
    gamma = sigma**2 / alpha
    base = ModelDenoiser(lambda x, a: x)
    guided = MMPSDenoiser(base, process.schedule, operator, observation, 0.0015, 12, "cg", covariance)
    plain = MMPSDenoiser(base, process.schedule, operator, observation, 0.0015, 12, "cg")
    assert not torch.allclose(guided(noisy, None, level), plain(noisy, None, level), atol=1e-3)
