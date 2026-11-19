import argparse

import torch

from spheridiff.common.config import PROJECT_ROOT
from spheridiff.smoke.runner import run_smoke
from spheridiff.smoke.stages import STAGES

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Run the smoke test of one model.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    return parser


def main():
    arguments = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = run_smoke(
        arguments.experiment,
        device,
        arguments.runs_root,
        arguments.cache_root,
        tuple(arguments.stages),
    )
    print("smoke test passed" if report["passed"] else "smoke test has failures")


if __name__ == "__main__":
    main()
