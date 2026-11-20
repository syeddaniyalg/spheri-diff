import json
import os
import subprocess
import sys

import pytest
import torch

from pipeline import TINY_SPHERIDIFF, TINY_WIND, write_experiment
from spheridiff.analysis.cooldown import cooldown_rows, sufficient_budget, write_cooldown_summary
from spheridiff.common.config import PROJECT_ROOT
from spheridiff.experiments.build import build_trainer
from spheridiff.experiments.spec import load_experiment
from spheridiff.kaggle.jobs import cooldown_job
from spheridiff.training.cooldown import (
    branch_finished,
    branch_name,
    cooldown_length,
    cooldown_settings,
    milestone_path,
)
from spheridiff.training.runlog import RunPaths, append_record, read_records
from spheridiff.training.schedule import cooldown_rate_at, learning_rate_at, scheduled_rate
from spheridiff.training.settings import load_train_settings

CPU = torch.device("cpu")


def cpu_environment():
    return {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
    }


def test_cooldown_rate_falls_from_the_peak_to_the_minimum():
    values = [cooldown_rate_at(step, 100, 120, 2e-4, 1e-6) for step in range(100, 121)]
    assert values[0] == pytest.approx(2e-4)
    assert values[-1] == pytest.approx(1e-6)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))
    assert values[5] == pytest.approx(1e-6 + (2e-4 - 1e-6) * (1 - 0.5))


def test_cooldown_settings_start_on_the_cosine_schedule():
    base = load_train_settings("pilot")
    settings = cooldown_settings(base, 22500, cooldown_length(22500, 0.15))
    assert settings.steps == 22500 + 3375
    assert settings.evaluate_every == settings.steps and settings.checkpoint_every == 0
    assert settings.milestones == () and settings.cooldown_start == 22500
    cosine = learning_rate_at(22500, 45000, base.learning_rate, base.min_learning_rate, base.warmup_steps)
    assert scheduled_rate(settings, 22500) == pytest.approx(cosine)
    assert scheduled_rate(settings, settings.steps) == pytest.approx(base.min_learning_rate)
    assert scheduled_rate(base, 1000) == learning_rate_at(1000, 45000, 3e-4, 1e-6, base.warmup_steps)


def test_cooldown_settings_reject_starts_outside_the_run():
    base = load_train_settings("pilot")
    for start in (0, 45000, 50000):
        with pytest.raises(ValueError):
            cooldown_settings(base, start, 100)
    with pytest.raises(ValueError):
        cooldown_length(100, 0.0)


def test_branch_names_and_milestone_paths(tmp_path):
    assert branch_name("pilot_wind_s", 11250) == "pilot_wind_s_cooldown_011250"
    assert milestone_path(tmp_path, "run", 22500) == tmp_path / "run" / "milestone_022500.pt"


@pytest.mark.parametrize("model", [TINY_WIND, TINY_SPHERIDIFF], ids=["wind", "spheridiff"])
def test_a_branch_continues_from_the_milestone_with_the_cooldown_rate(tmp_path, model):
    experiment, cache_root = write_experiment(tmp_path, model, steps=8, milestones=(0.5,))
    spec = load_experiment(experiment)
    base = load_train_settings(spec.train)
    full, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    full.run(resume=False)
    source = full.paths.root / "milestone_000004.pt"
    settings = cooldown_settings(base, 4, 2)
    branch, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, settings, branch_name("tiny", 4))
    branch.run(resume=True, start_from=source)
    steps = read_records(branch.paths.steps)
    assert [record["step"] for record in steps] == [5, 6]
    assert steps[0]["learning_rate"] == pytest.approx(settings.cooldown_peak)
    assert steps[1]["learning_rate"] < steps[0]["learning_rate"]
    evaluations = read_records(branch.paths.evaluations)
    assert [record["step"] for record in evaluations] == [6]
    assert branch_finished(branch.paths, 6)
    assert not branch_finished(RunPaths(tmp_path / "runs", "missing"), 6)


def test_a_branch_matches_the_steps_of_the_full_run_at_its_start(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=8, milestones=(0.5,))
    spec = load_experiment(experiment)
    base = load_train_settings(spec.train)
    full, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root)
    full.run(resume=False)
    settings = cooldown_settings(base, 4, 1)
    assert settings.cooldown_peak == pytest.approx(scheduled_rate(base, 4))
    branch, _ = build_trainer(spec, CPU, tmp_path / "runs", cache_root, settings, "branch")
    branch.run(resume=False, start_from=full.paths.root / "milestone_000004.pt")
    full_losses = {record["step"]: record["loss"] for record in read_records(full.paths.steps)}
    branch_losses = {record["step"]: record["loss"] for record in read_records(branch.paths.steps)}
    assert branch_losses[5] == pytest.approx(full_losses[5], abs=1e-6)


def test_the_cooldown_script_runs_every_branch_and_skips_finished_ones(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=8, milestones=(0.25, 0.5))
    spec = load_experiment(experiment)
    full, _ = build_trainer(spec, CPU, tmp_path / "pilot", cache_root)
    full.run(resume=False)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "cooldown.py"),
        "--experiment", experiment,
        "--milestones-root", str(tmp_path / "pilot"),
        "--starts", "4", "2",
        "--fraction", "0.5",
        "--keep-only-weights",
        "--runs-root", str(tmp_path / "runs"),
        "--cache-root", str(cache_root),
    ]
    first = subprocess.run(command, capture_output=True, text=True, env=cpu_environment())
    assert first.returncode == 0, first.stderr
    for start, end in ((2, 3), (4, 6)):
        paths = RunPaths(tmp_path / "runs", branch_name("tiny", start))
        assert branch_finished(paths, end)
        assert paths.weights.exists() and not paths.checkpoint.exists() and not paths.final.exists()
        snapshot = json.loads(paths.snapshot.read_text())
        assert snapshot["cooldown"]["start_step"] == start and snapshot["cooldown"]["end_step"] == end
    second = subprocess.run(command, capture_output=True, text=True, env=cpu_environment())
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("already finished") == 2


def test_the_cooldown_script_rejects_a_missing_milestone(tmp_path):
    experiment, cache_root = write_experiment(tmp_path, TINY_WIND, steps=4)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "cooldown.py"),
        "--experiment", experiment,
        "--milestones-root", str(tmp_path / "pilot"),
        "--starts", "2",
        "--runs-root", str(tmp_path / "runs"),
        "--cache-root", str(cache_root),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=cpu_environment())
    assert result.returncode != 0 and "does not exist" in result.stderr


def write_run(root, name, records, errors):
    paths = RunPaths(root, name)
    for step, value in records:
        append_record(paths.evaluations, {"step": step, "nrmse/mean": value, "crps_normalised/mean": value / 2, "val_loss/mean": value / 100})
    for step, scale in errors:
        append_record(paths.windows, {"step": step, "indices": [0, 1, 2, 3], "nmse": [scale * value for value in (0.9, 1.0, 1.1, 1.2)]})


def test_the_summary_finds_the_smallest_matching_budget(tmp_path):
    reference, runs = tmp_path / "pilot", tmp_path / "runs"
    write_run(reference, "full", [(0, 1.0), (10, 0.5), (20, 0.4), (40, 0.3)], [(40, 0.09)])
    write_run(runs, branch_name("full", 10), [(12, 0.45)], [(12, 0.2)])
    write_run(runs, branch_name("full", 20), [(24, 0.3)], [(24, 0.09)])
    rows = cooldown_rows(runs, reference, ["full"], [10, 20], tolerance=0.002)
    assert [row["matches_full"] for row in rows] == [False, True]
    assert rows[0]["milestone"]["nrmse/mean"] == 0.5 and rows[0]["full"]["nrmse/mean"] == 0.3
    assert sufficient_budget(rows, "full") == 24
    text = write_cooldown_summary(runs, reference, ["full"], [10, 20], 0.002)
    assert "smallest budget that matches the full run: step 24" in text
    assert (runs / "cooldown_summary.json").exists()


def test_the_cooldown_job_passes_its_arguments(tmp_path):
    job = cooldown_job("pilot_wind_s", tmp_path, [11250, 22500], 0.15)
    assert job.script == "cooldown.py" and not job.budgeted
    assert job.arguments[job.arguments.index("--starts") + 1 : job.arguments.index("--fraction")] == ("11250", "22500")
    assert "--keep-only-weights" in job.arguments
