import argparse
from pathlib import Path

from spheridiff.analysis.curves import plot_evaluations, plot_training_loss
from spheridiff.common.config import PROJECT_ROOT

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Plot the logged curves of several runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--references", nargs="*", default=[])
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--loss-window", type=int, default=200)
    return parser


def main():
    arguments = build_parser().parse_args()
    output = Path(arguments.output_dir or arguments.runs_root)
    output.mkdir(parents=True, exist_ok=True)
    plot_evaluations(arguments.runs_root, arguments.runs, arguments.references, output / "evaluation_curves.png")
    plot_training_loss(arguments.runs_root, arguments.runs, arguments.loss_window, output / "training_loss.png")
    print(f"saved plots to {output}")


if __name__ == "__main__":
    main()
