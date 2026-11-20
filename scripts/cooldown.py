import argparse
import dataclasses

import torch

from spheridiff.common.config import PROJECT_ROOT
from spheridiff.experiments.build import build_trainer, run_metadata
from spheridiff.experiments.spec import load_experiment
from spheridiff.training.cooldown import (
    branch_finished,
    branch_name,
    cooldown_length,
    cooldown_settings,
    milestone_path,
    prune_branch,
)
from spheridiff.training.loop import Trainer
from spheridiff.training.runlog import RunPaths
from spheridiff.training.settings import load_train_settings

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Cool down saved milestones of one experiment.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--milestones-root", required=True, help="folder that holds <experiment>/milestone_*.pt")
    parser.add_argument("--starts", type=int, nargs="+", required=True)
    parser.add_argument("--fraction", type=float, default=0.15, help="cooldown length as a fraction of the start step")
    parser.add_argument("--length", type=int, default=0, help="fixed cooldown length in steps, overrides the fraction")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--fresh", action="store_true", help="restart every branch from its milestone")
    parser.add_argument("--keep-only-weights", action="store_true", help="delete the large checkpoints of a finished branch")
    return parser


def check_milestone(path, start_step):
    if not path.exists():
        raise FileNotFoundError(f"the milestone {path} does not exist")
    step = torch.load(path, map_location="cpu", weights_only=False)["step"]
    if step != start_step:
        raise ValueError(f"{path} holds step {step}, not {start_step}")


def branch_metadata(spec, settings, components, device, source):
    metadata = run_metadata(spec, settings, components, device)
    metadata["cooldown"] = {
        "start_step": settings.cooldown_start,
        "end_step": settings.steps,
        "start_learning_rate": settings.cooldown_peak,
        "source": str(source),
    }
    return metadata


def branch_trainer(spec, settings, name, device, arguments, shared):
    if shared is None:
        return build_trainer(spec, device, arguments.runs_root, arguments.cache_root, settings, name)
    parts, components = shared
    paths = RunPaths(arguments.runs_root, name)
    return Trainer(dataclasses.replace(parts), settings, paths, device, spec.seed, name), components


def describe_branch(name, settings):
    return (
        f"{name}: steps {settings.cooldown_start} to {settings.steps}, "
        f"learning rate {settings.cooldown_peak:.3e} down to {settings.min_learning_rate:.1e}"
    )


def run_branch(spec, base, start_step, device, arguments, shared):
    length = arguments.length or cooldown_length(start_step, arguments.fraction)
    settings = cooldown_settings(base, start_step, length)
    name = branch_name(spec.name, start_step)
    if not arguments.fresh and branch_finished(RunPaths(arguments.runs_root, name), settings.steps):
        print(f"{name} is already finished at step {settings.steps}, skipping")
        return shared
    source = milestone_path(arguments.milestones_root, spec.name, start_step)
    check_milestone(source, start_step)
    trainer, components = branch_trainer(spec, settings, name, device, arguments, shared)
    trainer.parts.metadata = branch_metadata(spec, settings, components, device, source)
    print(describe_branch(name, settings))
    trainer.run(resume=not arguments.fresh, start_from=source)
    if arguments.keep_only_weights:
        prune_branch(trainer.paths)
    return trainer.parts, components


def main():
    arguments = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_experiment(arguments.experiment)
    base = load_train_settings(spec.train)
    shared = None
    for start_step in sorted(arguments.starts):
        shared = run_branch(spec, base, start_step, device, arguments, shared)


if __name__ == "__main__":
    main()
