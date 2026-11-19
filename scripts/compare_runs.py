import argparse

from spheridiff.analysis.comparison import compare_runs, format_comparison, write_comparison
from spheridiff.common.config import PROJECT_ROOT

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Compare the learning curve of a run with reference runs.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    return parser


def main():
    arguments = build_parser().parse_args()
    for reference in arguments.references:
        result = compare_runs(arguments.runs_root, arguments.run, reference)
        path = write_comparison(arguments.runs_root, result)
        print(format_comparison(result))
        print(f"  saved to {path}\n")


if __name__ == "__main__":
    main()
