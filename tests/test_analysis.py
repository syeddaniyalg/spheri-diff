import numpy as np

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
