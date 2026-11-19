import json
import shutil
from pathlib import Path

from spheridiff.analysis.comparison import compare_runs, format_comparison, write_comparison
from spheridiff.analysis.curves import plot_evaluations, plot_training_loss
from spheridiff.analysis.final_table import write_final_summary
from spheridiff.analysis.package import package_runs
from spheridiff.training.runlog import RunPaths, read_records

LOSS_WINDOW = 50
LOG_FILES = ("evaluations.jsonl", "steps.jsonl", "windows.jsonl")


def write_status(runs_root, outcomes):
    path = Path(runs_root) / "status.json"
    path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    return path


def copy_if_present(source, target):
    if Path(source).exists():
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


def stage_for_plots(runs_root, names, reference_root, reference_names):
    stage = Path(runs_root).parent / "_plot_stage"
    for root, group in ((runs_root, names), (reference_root, reference_names)):
        for name in group:
            for file in LOG_FILES:
                copy_if_present(Path(root) / name / file, stage / name / file)
    return stage


def runs_with_records(runs_root, names):
    return [name for name in names if read_records(RunPaths(runs_root, name).steps)]


def draw_figures(runs_root, names, references, reference_root=None, reference_names=()):
    plot_root = runs_root
    if reference_root is not None:
        plot_root = stage_for_plots(runs_root, names, reference_root, reference_names)
    present = runs_with_records(plot_root, [*names, *reference_names])
    if not present:
        return
    plot_evaluations(plot_root, present, references, Path(runs_root) / "evaluation_curves.png")
    plot_training_loss(plot_root, present, LOSS_WINDOW, Path(runs_root) / "training_loss.png")


def write_comparisons(runs_root, comparisons, reference_root=None, at_final=False):
    texts = []
    for run, reference in comparisons:
        result = compare_runs(runs_root, run, reference, reference_root=reference_root, at_final=at_final)
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


def final_groups(runs_root, final_names, reference_root, reference_names):
    groups = [(runs_root, list(final_names))]
    if reference_root is not None and reference_names:
        groups.append((reference_root, list(reference_names)))
    return groups


def finalise(
    runs_root,
    archive_path,
    outcomes,
    names,
    references=(),
    comparisons=(),
    reference_root=None,
    at_final=False,
    final_names=(),
    final_reference_names=(),
):
    Path(runs_root).mkdir(parents=True, exist_ok=True)
    write_status(runs_root, outcomes)
    reference_names = sorted({reference for _, reference in comparisons}) if reference_root else []
    guarded(
        runs_root,
        "figure",
        lambda: draw_figures(runs_root, names, list(references), reference_root, reference_names),
    )
    guarded(
        runs_root,
        "comparison",
        lambda: write_comparisons(runs_root, comparisons, reference_root, at_final),
    )
    guarded(
        runs_root,
        "final_summary",
        lambda: write_final_summary(
            runs_root, final_groups(runs_root, final_names, reference_root, final_reference_names)
        ),
    )
    return refresh_archive(runs_root, archive_path)
