from dataclasses import dataclass

from spheridiff.common.config import read_yaml
from spheridiff.common.seed import DEFAULT_SEED


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    data: str
    model: str
    train: str
    evaluation: str
    seed: int = DEFAULT_SEED
    pilot_train: str = "pilot"
    pretrained: str = None
    adaptation: str = "full"


def load_experiment(value):
    return ExperimentSpec(**read_yaml(value, "experiments"))
