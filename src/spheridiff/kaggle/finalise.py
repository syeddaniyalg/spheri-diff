import json
import shutil
from pathlib import Path

from spheridiff.analysis.curves import plot_evaluations, plot_training_loss
from spheridiff.analysis.package import package_runs
from spheridiff.training.runlog import RunPaths, read_records

LOSS_WINDOW = 50


def write_status(runs_root, outcomes):
    path = Path(runs_root) / "status.json"
    path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    return path


def runs_with_records(runs_root, names):
    return [name for name in names if read_records(RunPaths(runs_root, name).steps)]


def draw_figures(runs_root, names, references):
    present = runs_with_records(runs_root, names)
    if not present:
        return
    plot_evaluations(runs_root, present, references, Path(runs_root) / "evaluation_curves.png")
    plot_training_loss(runs_root, present, LOSS_WINDOW, Path(runs_root) / "training_loss.png")


def refresh_archive(runs_root, archive_path):
    return package_runs(runs_root, archive_path)


def finalise(runs_root, archive_path, outcomes, names, references=()):
    Path(runs_root).mkdir(parents=True, exist_ok=True)
    write_status(runs_root, outcomes)
    try:
        draw_figures(runs_root, names, list(references))
    except Exception as error:
        (Path(runs_root) / "figure_error.txt").write_text(repr(error), encoding="utf-8")
    return refresh_archive(runs_root, archive_path)


def copy_if_present(source, target):
    if Path(source).exists():
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
