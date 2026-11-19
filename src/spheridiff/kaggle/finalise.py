import json
import shutil
from pathlib import Path

from spheridiff.analysis.comparison import compare_runs, format_comparison, write_comparison
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


def write_comparisons(runs_root, comparisons):
    texts = []
    for run, reference in comparisons:
        result = compare_runs(runs_root, run, reference)
        write_comparison(runs_root, result)
        texts.append(format_comparison(result))
    (Path(runs_root) / "comparison_summary.txt").write_text("\n\n".join(texts), encoding="utf-8")


def guarded(runs_root, name, action):
    try:
        action()
    except Exception as error:
        (Path(runs_root) / f"{name}_error.txt").write_text(repr(error), encoding="utf-8")


def refresh_archive(runs_root, archive_path):
    return package_runs(runs_root, archive_path)


def finalise(runs_root, archive_path, outcomes, names, references=(), comparisons=()):
    Path(runs_root).mkdir(parents=True, exist_ok=True)
    write_status(runs_root, outcomes)
    guarded(runs_root, "figure", lambda: draw_figures(runs_root, names, list(references)))
    guarded(runs_root, "comparison", lambda: write_comparisons(runs_root, comparisons))
    return refresh_archive(runs_root, archive_path)


def copy_if_present(source, target):
    if Path(source).exists():
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
