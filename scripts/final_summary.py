import argparse

from spheridiff.analysis.final_table import write_final_summary
from spheridiff.common.config import PROJECT_ROOT

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Tabulate final evaluations of several runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--references", nargs="*", default=[])
    return parser


def main():
    arguments = build_parser().parse_args()
    groups = [(arguments.runs_root, arguments.runs)]
    if arguments.reference_root:
        groups.append((arguments.reference_root, arguments.references))
    text = write_final_summary(arguments.runs_root, groups)
    print(text or "no final evaluations were found")


if __name__ == "__main__":
    main()
