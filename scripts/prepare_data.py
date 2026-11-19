import argparse
import sys

from spheridiff.common.config import load_data_config
from spheridiff.data.plan import check_cache, plan_lines


def run_plan(config, arguments):
    print("\n".join(plan_lines(config)))
    return 0


def run_download(config, arguments):
    from spheridiff.data.download import download_all

    download_all(config, arguments.years)
    return 0


def run_statistics(config, arguments):
    from spheridiff.data.statistics import build_statistics

    build_statistics(config)
    print("statistics saved")
    return 0


def run_check(config, arguments):
    problems = check_cache(config)
    print("\n".join(problems) if problems else "cache is complete")
    return 1 if problems else 0


COMMANDS = {
    "plan": run_plan,
    "download": run_download,
    "statistics": run_statistics,
    "check": run_check,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Build and check the local data cache.")
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument(
        "--config",
        required=True,
        help="path to a data config, or the name of a file in configs/data (without .yaml)",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="folder of the cache, used instead of the one in the config",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="download only these years (default: every year in the splits)",
    )
    return parser


def main():
    arguments = build_parser().parse_args()
    config = load_data_config(arguments.config, arguments.cache_root)
    sys.exit(COMMANDS[arguments.command](config, arguments))


if __name__ == "__main__":
    main()
