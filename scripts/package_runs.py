import argparse

from spheridiff.analysis.package import package_runs
from spheridiff.common.config import PROJECT_ROOT

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Pack all run folders into one zip archive.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output", required=True)
    return parser


def main():
    arguments = build_parser().parse_args()
    archive = package_runs(arguments.runs_root, arguments.output)
    print(f"wrote {archive} ({archive.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
