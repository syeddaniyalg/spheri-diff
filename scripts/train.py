import argparse
import dataclasses

import torch

from spheridiff.analysis.package import package_runs
from spheridiff.common.config import PROJECT_ROOT
from spheridiff.experiments.build import build_trainer
from spheridiff.experiments.spec import load_experiment
from spheridiff.training.settings import load_train_settings

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Train one experiment.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    parser.add_argument("--package", default=None, help="zip all runs into this file when training ends")
    parser.add_argument(
        "--max-seconds", type=int, default=0, help="stop with a checkpoint after this many seconds"
    )
    return parser


def main():
    arguments = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_experiment(arguments.experiment)
    settings = load_train_settings(spec.train)
    if arguments.max_seconds:
        settings = dataclasses.replace(settings, max_seconds=float(arguments.max_seconds))
    trainer, _ = build_trainer(spec, device, arguments.runs_root, arguments.cache_root, settings)
    trainer.run(resume=not arguments.fresh)
    if arguments.package:
        print(f"wrote {package_runs(arguments.runs_root, arguments.package)}")


if __name__ == "__main__":
    main()
