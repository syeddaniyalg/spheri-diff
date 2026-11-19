import numpy as np

from spheridiff.training.runlog import RunPaths, read_records

METRICS = ("val_loss/mean", "nrmse/mean", "crps_normalised/mean")
HORIZONTAL_AXES = (
    ("step", "training steps"),
    ("samples_seen", "samples seen"),
    ("train_seconds", "training seconds"),
)


def load_series(runs_root, name, key, axis):
    records = [record for record in read_records(RunPaths(runs_root, name).evaluations) if key in record]
    return np.array([record[axis] for record in records]), np.array([record[key] for record in records])


def first_crossing(runs_root, name, key, level, axis="step"):
    positions, values = load_series(runs_root, name, key, axis)
    reached = np.flatnonzero(values <= level)
    return None if len(reached) == 0 else float(positions[reached[0]])


def moving_average(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_evaluations(runs_root, names, references, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(METRICS), len(HORIZONTAL_AXES), figsize=(15, 3.2 * len(METRICS)))
    for row, key in enumerate(METRICS):
        for column, (axis, label) in enumerate(HORIZONTAL_AXES):
            draw_metric(axes[row, column], runs_root, names, references, key, axis)
            axes[row, column].set_xlabel(label)
            axes[row, column].set_ylabel(key)
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)


def draw_metric(axis_object, runs_root, names, references, key, axis):
    for name in names:
        positions, values = load_series(runs_root, name, key, axis)
        axis_object.plot(positions, values, marker="o", markersize=3, label=name)
    for name in references:
        _, values = load_series(runs_root, name, key, axis)
        if len(values):
            axis_object.axhline(values[-1], linestyle="--", linewidth=0.8, label=f"{name} final")


def plot_training_loss(runs_root, names, window, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis_object = plt.subplots(figsize=(8, 4))
    for name in names:
        records = read_records(RunPaths(runs_root, name).steps)
        losses = np.array([record["loss"] for record in records])
        width = max(1, min(window, len(losses)))
        axis_object.plot(moving_average(losses, width), label=name)
    axis_object.set_xlabel("training steps")
    axis_object.set_ylabel(f"training loss (moving average of {window})")
    axis_object.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)
