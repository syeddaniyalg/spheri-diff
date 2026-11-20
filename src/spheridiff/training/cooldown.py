import dataclasses
import json
from pathlib import Path

from .runlog import milestone_name
from .schedule import learning_rate_at

PRUNED_FILES = ("checkpoint.pt", "final.pt")


def cooldown_length(start_step, fraction):
    if fraction <= 0:
        raise ValueError("the cooldown fraction must be positive")
    return max(1, round(fraction * start_step))


def cooldown_settings(settings, start_step, length):
    if not 0 < start_step < settings.steps:
        raise ValueError(f"the cooldown must start inside the run, not at step {start_step}")
    end = start_step + length
    peak = learning_rate_at(
        start_step, settings.steps, settings.learning_rate, settings.min_learning_rate, settings.warmup_steps
    )
    return dataclasses.replace(
        settings,
        steps=end,
        evaluate_every=end,
        checkpoint_every=0,
        milestones=(),
        max_seconds=0.0,
        cooldown_start=start_step,
        cooldown_peak=peak,
    )


def branch_name(experiment, start_step):
    return f"{experiment}_cooldown_{start_step:06d}"


def milestone_path(milestones_root, experiment, start_step):
    return Path(milestones_root) / experiment / milestone_name(start_step)


def branch_finished(paths, end_step):
    if not paths.summary.exists():
        return False
    return json.loads(paths.summary.read_text(encoding="utf-8")).get("steps", 0) >= end_step


def prune_branch(paths):
    for name in PRUNED_FILES:
        (paths.root / name).unlink(missing_ok=True)
