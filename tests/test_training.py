import dataclasses
import json
import math
import zipfile

import pytest
import torch

from pipeline import TINY_SPHERIDIFF, TINY_WIND, write_experiment
from spheridiff.analysis.package import package_runs
from spheridiff.common.seed import DEFAULT_SEED
from spheridiff.experiments.build import build_components, build_full_evaluation, build_trainer
from spheridiff.experiments.spec import load_experiment
from spheridiff.evaluation.settings import TASKS, load_full_settings
from spheridiff.models.registry import count_parameters
from spheridiff.smoke.curves import edge_means, loss_ratio, skipped_fraction
from spheridiff.smoke.estimate import estimate_pilot_time
from spheridiff.training.ema import ExponentialMovingAverage
from spheridiff.training.runlog import read_records
from spheridiff.training.schedule import learning_rate_at
from spheridiff.training.settings import load_train_settings

CPU = torch.device("cpu")
MODELS = {"wind": TINY_WIND, "spheridiff": TINY_SPHERIDIFF}


def test_learning_rate_warms_up_then_decays():
    values = [learning_rate_at(step, 100, 1e-3, 1e-5, 10) for step in range(100)]
    assert values[0] == pytest.approx(1e-4)
    assert values[9] == pytest.approx(1e-3)
    assert values[10] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-5, abs=2e-6)
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values[10:], values[11:]))


def test_ema_follows_the_parameters():
    model = torch.nn.Linear(2, 2)
    average = ExponentialMovingAverage(model, decay=0.5)
    start = model.weight.detach().clone()
    model.weight.data.add_(1.0)
    average.update(model)
    assert torch.allclose(average.shadow["weight"], start + 0.9, atol=1e-6)


def test_ema_applied_restores_the_weights():
    model = torch.nn.Linear(2, 2)
    average = ExponentialMovingAverage(model, decay=0.9)
    original = model.weight.detach().clone()
    model.weight.data.add_(1.0)
    average.update(model)
    trained = model.weight.detach().clone()
    with average.applied(model):
        assert torch.allclose(model.weight, average.shadow["weight"])
    assert torch.equal(model.weight, trained)
    assert not torch.equal(trained, original)


@pytest.mark.parametrize("kind", ["wind", "spheridiff"])
def test_training_reduces_the_loss_on_fixed_windows(tmp_path, kind):
    experiment, cache_root = write_experiment(
        tmp_path, MODELS[kind], steps=150, evaluate_every=50, fixed_windows=4
    )
    spec = load_experiment(experiment)
    trainer, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    steps = read_records(trainer.paths.steps)
    assert len(steps) == 150
    assert loss_ratio(steps) < 0.85
    assert all(math.isfinite(record["loss"]) for record in steps)
    assert skipped_fraction(steps) == 0.0
    evaluations = read_records(trainer.paths.evaluations)
    assert [record["step"] for record in evaluations] == [0, 50, 100, 150]
    assert "baseline/mean/nrmse_mean" in evaluations[0]
    assert "baseline/persistence/nrmse_mean" in evaluations[0]
    assert evaluations[-1]["nrmse/mean"] < evaluations[0]["baseline/mean/nrmse_mean"]


def test_every_step_is_logged_with_timing_fields(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=5)
    trainer, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    record = read_records(trainer.paths.steps)[-1]
    expected = {"step", "samples_seen", "loss", "learning_rate", "grad_norm", "skipped", "loss_scale"}
    expected |= {"step_seconds", "data_seconds", "samples_per_second", "train_seconds", "memory_gb"}
    assert expected <= set(record)
    assert record["samples_seen"] == 5 * 4


@pytest.mark.parametrize("kind", ["wind", "spheridiff"])
def test_resuming_matches_an_uninterrupted_run(tmp_path, kind):
    experiment, cache_root = write_experiment(tmp_path, MODELS[kind], steps=6)
    spec = load_experiment(experiment)
    settings = dataclasses.replace(load_train_settings(experiment_train(experiment)), checkpoint_every=3)
    continuous, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, settings, "a")
    continuous.run(resume=False)
    first, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, settings, "b")
    first.stop_at = 3
    first.run(resume=False)
    second, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, settings, "b")
    second.run(resume=True)
    assert second.step == 6
    for left, right in zip(continuous.model.parameters(), second.model.parameters()):
        assert torch.allclose(left, right, atol=1e-6)
    losses_a = [r["loss"] for r in read_records(continuous.paths.steps)]
    losses_b = [r["loss"] for r in read_records(second.paths.steps)]
    assert losses_a == pytest.approx(losses_b, abs=1e-6)


def experiment_train(experiment_path):
    return load_experiment(experiment_path).train


def test_interval_checkpoint_is_updated_and_a_final_one_is_kept(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=5, checkpoint_every=2)
    spec = load_experiment(experiment)
    trainer, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    trainer.stop_at = 4
    trainer.run(resume=False)
    assert torch.load(trainer.paths.checkpoint, weights_only=False)["step"] == 4
    assert not trainer.paths.final.exists()
    resumed, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    resumed.run(resume=True)
    assert torch.load(resumed.paths.checkpoint, weights_only=False)["step"] == 5
    assert torch.load(resumed.paths.final, weights_only=False)["step"] == 5
    assert torch.load(resumed.paths.weights, weights_only=False)["step"] == 5


def test_milestone_checkpoints_are_kept_and_can_resume(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=8, milestones=(0.25, 0.5, 0.75))
    spec = load_experiment(experiment)
    trainer, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    kept = sorted(path.name for path in trainer.paths.root.glob("milestone_*.pt"))
    assert kept == ["milestone_000002.pt", "milestone_000004.pt", "milestone_000006.pt"]
    state = torch.load(trainer.paths.root / "milestone_000004.pt", weights_only=False)
    assert state["step"] == 4 and {"model", "optimizer", "ema", "scaler"} <= set(state)


def test_milestone_steps_ignore_the_ends():
    settings = load_train_settings("pilot")
    assert settings.milestone_steps == {11250, 22500, 33750}
    assert dataclasses.replace(settings, milestones=(0.0, 1.0, 1.5)).milestone_steps == set()


def test_per_window_scores_are_logged_at_every_evaluation(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_SPHERIDIFF, steps=4, evaluate_every=2)
    trainer, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    records = read_records(trainer.paths.windows)
    assert [record["step"] for record in records] == [0, 2, 4]
    for record in records:
        assert len(record["indices"]) == len(record["nmse"]) == len(record["ncrps"]) == 4
        assert all(math.isfinite(value) and value >= 0 for value in record["nmse"] + record["ncrps"])
    assert records[0]["indices"] == records[-1]["indices"]
    mean_error = math.sqrt(sum(records[-1]["nmse"]) / 4)
    assert mean_error == pytest.approx(
        read_records(trainer.paths.evaluations)[-1]["nrmse/mean"], rel=0.35
    )


def test_resuming_truncates_the_window_log(tmp_path):
    experiment, cache_root = write_experiment(
        tmp_path, TINY_WIND, steps=6, evaluate_every=2, checkpoint_every=2
    )
    spec = load_experiment(experiment)
    first, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    first.stop_at = 2
    first.run(resume=False)
    second, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    second.run(resume=True)
    steps = [record["step"] for record in read_records(second.paths.windows)]
    assert steps == [0, 2, 4, 6]


def test_the_time_limit_stops_training_with_a_resumable_checkpoint(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=100000)
    spec = load_experiment(experiment)
    settings = dataclasses.replace(load_train_settings(spec.train), max_seconds=1.0)
    trainer, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, settings)
    trainer.run(resume=False)
    assert 0 < trainer.step < 100000
    assert torch.load(trainer.paths.checkpoint, weights_only=False)["step"] == trainer.step
    assert not trainer.paths.final.exists()
    assert trainer.paths.summary.exists() and trainer.paths.weights.exists()
    resumed, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, dataclasses.replace(settings, steps=trainer.step + 2))
    resumed.run(resume=True)
    assert resumed.step == trainer.step + 2


def test_a_config_snapshot_is_written(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_SPHERIDIFF, steps=2)
    trainer, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    snapshot = json.loads(trainer.paths.snapshot.read_text())
    assert {"experiment", "train", "model", "parameters", "data", "torch", "device"} <= set(snapshot)
    assert snapshot["model"]["name"] == "spheridiff"
    assert snapshot["train"]["steps"] == 2
    assert snapshot["parameters"] > 0


def test_run_folders_are_packed_into_one_archive(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=2)
    trainer, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    archive = package_runs(tmp_path / "runs", tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    expected = {"steps.jsonl", "final.pt", "ema_weights.pt", "summary.json", "config_snapshot.json"}
    assert {f"tiny/{name}" for name in expected} <= names


def test_the_stored_ema_weights_load_back_into_a_model(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_SPHERIDIFF, steps=4)
    spec = load_experiment(experiment)
    trainer, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    fresh = build_components(spec, cache_root).model
    fresh.load_state_dict(torch.load(trainer.paths.weights, weights_only=False)["model"])
    for name, parameter in fresh.named_parameters():
        assert torch.allclose(parameter, trainer.ema.shadow[name])


@pytest.mark.parametrize("kind", ["wind", "spheridiff"])
def test_full_evaluation_runs_every_path(tmp_path, kind):
    experiment, cache_root = write_experiment(tmp_path, MODELS[kind], steps=2)
    spec = load_experiment(experiment)
    components = build_components(spec, cache_root)
    evaluation = build_full_evaluation(components, load_full_settings(spec.evaluation), CPU)
    record = evaluation.run(components.model)
    assert {key.split("/")[0] for key in record} >= {"forecast", *TASKS}
    assert all(math.isfinite(value) for value in record.values())
    assert "forecast/ssr/500_temperature/24h" in record
    assert "forecast/rmse/500_temperature/48h" in record
    assert "spatial_downscaling/baseline/nrmse_mean" in record


def test_untrained_models_start_at_the_mean_baseline(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=2, evaluate_every=1)
    trainer, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "runs", cache_root)
    trainer.run(resume=False)
    first = read_records(trainer.paths.evaluations)[0]
    assert first["nrmse/mean"] == pytest.approx(first["baseline/mean/nrmse_mean"], rel=1e-4)


def test_curve_helpers():
    records = [{"loss": 2.0 - 0.01 * step, "skipped": step % 50 == 0} for step in range(100)]
    head, tail = edge_means([record["loss"] for record in records])
    assert head > tail
    assert 0.5 < loss_ratio(records) < 1.0
    assert skipped_fraction(records) == pytest.approx(0.02)


def test_pilot_time_estimate_scales_with_steps():
    steps = [{"samples_per_second": 100.0} for _ in range(30)]
    evaluations = [{"evaluation_seconds": 10.0} for _ in range(3)]
    smoke = load_train_settings("smoke")
    pilot = load_train_settings("pilot")
    estimate = estimate_pilot_time(steps, evaluations, smoke, pilot)
    assert estimate["training_hours"] == pytest.approx(pilot.steps * pilot.batch_size / 100.0 / 3600)
    assert estimate["total_hours_with_margin"] > estimate["total_hours"] > estimate["training_hours"]


def test_pilot_configs_are_consistent():
    pilot = load_train_settings("pilot")
    assert pilot.steps % pilot.evaluate_every == 0
    assert pilot.evaluation.members >= 2
    for name in ("pilot_wind_s", "pilot_wind_m", "pilot_wind_l", "pilot_spheridiff_b4"):
        spec = load_experiment(name)
        assert spec.train == "pilot" and spec.seed == DEFAULT_SEED


def test_model_sizes_follow_the_planned_ladder():
    from helpers import make_grid
    from spheridiff.models.context import ModelContext
    from spheridiff.models.registry import build_model
    from spheridiff.models.spec import load_model_spec

    context = ModelContext(25, 10, 5, make_grid(32, 64))
    sizes = {}
    for name in ("wind_s", "wind_m", "wind_l", "spheridiff_b4"):
        spec = load_model_spec(name)
        with torch.device("meta"):
            model = build_model(spec.name, spec.architecture, context)
        sizes[name] = count_parameters(model)
    assert sizes["wind_s"] < sizes["wind_m"] < sizes["wind_l"]
    assert sizes["spheridiff_b4"] < sizes["wind_s"]
    assert 2.9e6 < sizes["wind_s"] < 3.2e6
    assert 12.0e6 < sizes["wind_l"] < 12.5e6
