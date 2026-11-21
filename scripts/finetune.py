import argparse
import dataclasses
from pathlib import Path

import torch

from spheridiff.common.config import PROJECT_ROOT
from spheridiff.experiments.build import build_trainer
from spheridiff.experiments.spec import load_experiment
from spheridiff.finetune.transfer import pretrained_weights, transfer_weights
from spheridiff.training.runlog import RunPaths, save_atomic, write_summary
from spheridiff.training.settings import load_train_settings

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
DEFAULT_PRETRAINED_ROOT = PROJECT_ROOT / "results"
ZERO_SHOT_SUFFIX = "_zeroshot"


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained model on a regional grid.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pretrained-root", default=str(DEFAULT_PRETRAINED_ROOT))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    parser.add_argument("--max-seconds", type=int, default=0, help="stop with a checkpoint after this many seconds")
    return parser


def pretrained_path(spec, root):
    if spec.pretrained is None:
        raise SystemExit(f"experiment {spec.name} names no pretrained weights")
    path = Path(root) / spec.pretrained
    if not path.exists():
        raise SystemExit(f"pretrained weights {path} do not exist")
    return path


def save_zero_shot(model, runs_root, spec, source):
    paths = RunPaths(runs_root, spec.name + ZERO_SHOT_SUFFIX)
    weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    save_atomic(paths.weights, {"step": 0, "model": weights})
    write_summary(paths.snapshot, {"experiment": dataclasses.asdict(spec), "pretrained": str(source)})


def main():
    arguments = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_experiment(arguments.experiment)
    settings = load_train_settings(spec.train)
    if arguments.max_seconds:
        settings = dataclasses.replace(settings, max_seconds=float(arguments.max_seconds))
    source = pretrained_path(spec, arguments.pretrained_root)

    def prepare(model):
        transfer_weights(model, pretrained_weights(source))
        save_zero_shot(model, arguments.runs_root, spec, source)

    trainer, _ = build_trainer(spec, device, arguments.runs_root, arguments.cache_root, settings, prepare_model=prepare)
    print(f"fine-tuning {spec.name} from {source} ({spec.adaptation})")
    trainer.run(resume=not arguments.fresh)


if __name__ == "__main__":
    main()
