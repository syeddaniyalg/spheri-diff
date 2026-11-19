import json

import numpy as np
import pytest

from spheridiff.analysis.comparison import (
    compare_runs,
    curve_area,
    first_sustained,
    format_comparison,
    fraction_table,
    write_comparison,
)
from spheridiff.analysis.curves import (
    first_crossing,
    load_series,
    moving_average,
    plot_evaluations,
    plot_training_loss,
)
from spheridiff.training.runlog import RunPaths, append_record


def write_run(root, name, offset):
    paths = RunPaths(root, name)
    for step in range(1, 101):
        append_record(paths.steps, {"step": step, "loss": 1.0 / step + offset})
    for index, step in enumerate(range(0, 101, 25)):
        record = {"step": step, "samples_seen": step * 8, "train_seconds": step * 0.5}
        record.update({"val_loss/mean": 1.0 - 0.2 * index + offset, "nrmse/mean": 1.0 - 0.1 * index})
        record["crps_normalised/mean"] = 0.8 - 0.1 * index
        append_record(paths.evaluations, record)


def write_curve(root, name, values, windows=None, samples=16, seconds=1.0):
    paths = RunPaths(root, name)
    for index, value in enumerate(values):
        step = (index + 1) * 100
        record = {"step": step, "samples_seen": step * samples, "train_seconds": step * seconds}
        record.update({"nrmse/mean": value, "val_loss/mean": value, "crps_normalised/mean": value})
        append_record(paths.evaluations, record)
        if windows is not None:
            append_record(paths.windows, {"step": step, "indices": list(range(len(windows))), "nmse": windows})


def test_sustained_crossing_needs_two_evaluations():
    assert first_sustained([0.9, 0.5, 0.7, 0.4, 0.4], 0.5) == 3
    assert first_sustained([0.9, 0.5, 0.7, 0.4, 0.45], 0.5) == 3
    assert first_sustained([0.9, 0.8], 0.5) is None
    assert first_sustained([0.4], 0.5) == 0


def test_fractions_and_curve_area():
    records = [{"step": 100 * (index + 1), "v": 1.0 - 0.1 * index} for index in range(10)]
    table = fraction_table(records, "v")
    assert table["1.0"] == {"step": 1000, "value": pytest.approx(0.1)}
    assert table["0.5"]["step"] == 500
    assert table["0.05"]["step"] == 100
    assert curve_area(records, "v") == pytest.approx(0.55)


def test_a_faster_run_crosses_the_reference_target_early(tmp_path):
    write_curve(tmp_path, "slow", [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.4, 0.4, 0.4])
    write_curve(tmp_path, "fast", [0.7, 0.55, 0.45, 0.4, 0.38, 0.37, 0.36, 0.36, 0.35, 0.35], samples=8)
    result = compare_runs(tmp_path, "fast", "slow")
    assert result["target"] == pytest.approx(0.4)
    assert result["crossing"]["step"] == 400
    assert result["crossing"]["samples_seen"] == 400 * 8
    assert result["fractions"]["run"]["0.5"]["value"] < result["fractions"]["reference"]["0.5"]["value"]
    assert result["curve_area"]["run"] < result["curve_area"]["reference"]
    assert result["paired"] is None
    assert "reached at step 400" in format_comparison(result)
    path = write_comparison(tmp_path, result)
    assert json.loads(path.read_text())["run"] == "fast"


def test_a_run_that_never_reaches_the_target_is_reported(tmp_path):
    write_curve(tmp_path, "weak", [0.9, 0.8, 0.7, 0.6])
    write_curve(tmp_path, "strong", [0.5, 0.4, 0.3, 0.2])
    result = compare_runs(tmp_path, "weak", "strong")
    assert result["crossing"] is None
    assert "did not reach" in format_comparison(result)


def test_paired_bootstrap_separates_a_clear_difference(tmp_path):
    generator = np.random.default_rng(0)
    base = generator.uniform(0.2, 0.4, 40)
    write_curve(tmp_path, "better", [0.5, 0.4], windows=list(base * 0.64))
    write_curve(tmp_path, "worse", [0.6, 0.5], windows=list(base))
    paired = compare_runs(tmp_path, "better", "worse")["paired"]
    assert paired["step"] == 200 and paired["windows"] == 40
    assert paired["high"] < 0 < -paired["low"]
    assert paired["fraction_run_better"] > 0.99
    same = compare_runs(tmp_path, "worse", "worse")["paired"]
    assert same["difference"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0


def test_paired_bootstrap_needs_matching_windows(tmp_path):
    write_curve(tmp_path, "one", [0.5, 0.4], windows=[0.1, 0.2, 0.3])
    write_curve(tmp_path, "two", [0.5, 0.4], windows=[0.1, 0.2])
    assert compare_runs(tmp_path, "one", "two")["paired"] is None


def test_series_and_crossing(tmp_path):
    write_run(tmp_path, "slow", 0.2)
    positions, values = load_series(tmp_path, "slow", "val_loss/mean", "step")
    assert list(positions) == [0, 25, 50, 75, 100]
    assert np.isclose(values[0], 1.2)
    assert first_crossing(tmp_path, "slow", "nrmse/mean", 0.75) == 75.0
    assert first_crossing(tmp_path, "slow", "nrmse/mean", 0.01) is None
    assert first_crossing(tmp_path, "slow", "nrmse/mean", 0.75, "samples_seen") == 600.0


def test_moving_average_length():
    assert len(moving_average(np.arange(10.0), 3)) == 8


def test_plots_are_written(tmp_path):
    write_run(tmp_path, "first", 0.0)
    write_run(tmp_path, "second", 0.1)
    plot_evaluations(tmp_path, ["first"], ["second"], tmp_path / "curves.png")
    plot_training_loss(tmp_path, ["first", "second"], 20, tmp_path / "loss.png")
    assert (tmp_path / "curves.png").stat().st_size > 5000
    assert (tmp_path / "loss.png").stat().st_size > 5000
