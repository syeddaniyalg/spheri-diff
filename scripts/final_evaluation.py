import argparse

import torch
from tqdm import tqdm

from spheridiff.common.config import PROJECT_ROOT, load_data_config
from spheridiff.evaluation.protocol import load_protocol
from spheridiff.experiments.final import final_datasets, run_final_evaluation
from spheridiff.experiments.spec import load_experiment

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate trained runs under the fixed final protocol.")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--protocol", default="final_v1")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=None)
    return parser


def main():
    arguments = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_protocol(arguments.protocol)
    config = load_data_config(load_experiment(arguments.experiments[0]).data, arguments.cache_root)
    datasets = final_datasets(config, protocol)
    for experiment in tqdm(arguments.experiments, desc="final evaluation"):
        run_final_evaluation(experiment, protocol, arguments.runs_root, arguments.cache_root, device, datasets)
        tqdm.write(f"saved the final evaluation of {experiment}")


if __name__ == "__main__":
    main()
