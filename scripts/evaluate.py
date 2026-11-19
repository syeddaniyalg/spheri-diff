import argparse
import json

import torch

from spheridiff.common.config import PROJECT_ROOT
from spheridiff.evaluation.settings import TASKS, load_full_settings
from spheridiff.experiments.build import build_components, build_full_evaluation, load_weights
from spheridiff.experiments.spec import load_experiment
from spheridiff.training.runlog import RunPaths

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Run the full evaluation of a trained experiment.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--skip-forecast", action="store_true")
    return parser


def main():
    arguments = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_experiment(arguments.experiment)
    paths = RunPaths(arguments.runs_root, spec.name)
    components = build_components(spec, arguments.cache_root)
    load_weights(components.model, paths.weights, device)
    evaluation = build_full_evaluation(components, load_full_settings(spec.evaluation), device)
    record = evaluation.run(components.model, arguments.tasks, not arguments.skip_forecast)
    target = paths.root / "full_evaluation.json"
    target.write_text(json.dumps(record, indent=2))
    print(f"saved {len(record)} values to {target}")


if __name__ == "__main__":
    main()
