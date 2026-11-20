from dataclasses import dataclass

from spheridiff.common.config import read_yaml
from spheridiff.common.seed import DEFAULT_SEED

PRECISIONS = ("fp16", "bf16", "fp32")


@dataclass(frozen=True)
class EvaluationSettings:
    windows: int
    members: int
    ddim_steps: int
    batch_size: int
    levels: tuple
    fields: tuple
    seed: int = DEFAULT_SEED

    def __post_init__(self):
        if self.members < 2:
            raise ValueError("the unbiased CRPS needs at least two ensemble members")


@dataclass(frozen=True)
class TrainSettings:
    steps: int
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    warmup_fraction: float
    weight_decay: float
    gradient_clip: float
    ema_decay: float
    precision: str
    evaluate_every: int
    checkpoint_every: int
    source_levels: tuple
    evaluation: EvaluationSettings
    train_split: str = "train"
    fixed_windows: int = 0
    max_seconds: float = 0.0
    milestones: tuple = ()
    cooldown_start: int = 0
    cooldown_peak: float = 0.0

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")

    @property
    def warmup_steps(self):
        return int(self.warmup_fraction * self.steps)

    @property
    def milestone_steps(self):
        steps = {round(fraction * self.steps) for fraction in self.milestones}
        return {step for step in steps if 0 < step < self.steps}


def evaluation_from_dict(raw):
    values = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
    return EvaluationSettings(**values)


def train_settings_from_dict(raw):
    values = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
    values["evaluation"] = evaluation_from_dict(raw["evaluation"])
    return TrainSettings(**values)


def load_train_settings(value):
    return train_settings_from_dict(read_yaml(value, "train"))
