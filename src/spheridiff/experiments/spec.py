from dataclasses import dataclass

from spheridiff.common.config import read_yaml


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    data: str
    model: str
    train: str
    evaluation: str
    seed: int
    pilot_train: str = "pilot"


def load_experiment(value):
    return ExperimentSpec(**read_yaml(value, "experiments"))
