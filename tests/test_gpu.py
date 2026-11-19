import dataclasses
import math
import time

import pytest
import torch

from spheridiff.common.config import load_data_config
from spheridiff.experiments.build import build_trainer
from spheridiff.experiments.spec import load_experiment
from spheridiff.smoke.stages import static_stage, tasks_stage
from spheridiff.training.runlog import read_records
from spheridiff.training.settings import load_train_settings

CACHE_READY = (load_data_config("era5_5p6deg").cache_root / "stats.npz").exists()
READY = torch.cuda.is_available() and CACHE_READY
EXPERIMENTS = ["smoke_wind_s", "smoke_spheridiff_b4"]

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not READY, reason="needs a CUDA device and the local data cache"),
]


@pytest.fixture(autouse=True)
def release_memory():
    yield
    torch.cuda.empty_cache()


def tiny_settings(steps, evaluate_every=0):
    base = load_train_settings("overfit")
    evaluation = dataclasses.replace(base.evaluation, windows=4, members=2, ddim_steps=3, batch_size=4)
    return dataclasses.replace(
        base,
        steps=steps,
        batch_size=2,
        evaluate_every=evaluate_every,
        checkpoint_every=0,
        evaluation=evaluation,
    )


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_static_checks_pass_with_the_pilot_sizes(name):
    device = torch.device("cuda")
    result = static_stage(load_experiment(name), device, None, None)
    assert result["passed"]
    low, high = (2.9e6, 3.2e6) if "wind" in name else (1.4e6, 1.7e6)
    assert low < result["parameters"] < high


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_a_few_fp16_steps_train_the_pilot_sized_models(name, tmp_path):
    device = torch.device("cuda")
    trainer, _ = build_trainer(load_experiment(name), device, tmp_path, None, tiny_settings(12))
    started = time.perf_counter()
    trainer.run(resume=False)
    records = read_records(trainer.paths.steps)
    print(f"{name}: {time.perf_counter() - started:.1f} s, peak {records[-1]['memory_gb']:.2f} GB")
    assert len(records) == 12
    assert all(math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"]) for record in records)
    assert not all(record["skipped"] for record in records)
    assert records[-1]["memory_gb"] < 5.5


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_periodic_evaluation_runs_on_the_gpu(name, tmp_path):
    device = torch.device("cuda")
    settings = tiny_settings(3, evaluate_every=3)
    trainer, _ = build_trainer(load_experiment(name), device, tmp_path, None, settings)
    trainer.run(resume=False)
    evaluations = read_records(trainer.paths.evaluations)
    assert [record["step"] for record in evaluations] == [0, 3]
    first = evaluations[0]
    assert "baseline/mean/nrmse_mean" in first and "baseline/persistence/nrmse_mean" in first
    assert first["nrmse/mean"] == pytest.approx(first["baseline/mean/nrmse_mean"], rel=1e-3)
    assert all(math.isfinite(value) for record in evaluations for value in record.values())


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_evaluation_path_runs_on_the_gpu(name):
    device = torch.device("cuda")
    result = tasks_stage(load_experiment(name), device, None, None)
    assert result["passed"], result
