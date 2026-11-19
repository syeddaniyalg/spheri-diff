from dataclasses import dataclass

from spheridiff.common.config import read_yaml
from spheridiff.common.seed import DEFAULT_SEED

TASKS = ("spatial_downscaling", "temporal_downscaling", "sparse_reconstruction")


@dataclass(frozen=True)
class FullEvaluationSettings:
    split: str
    windows: int
    members: int
    ddim_steps: int
    batch_size: int
    forecast_frames: int
    spatial_factor: int
    sparse_fraction: float
    noise_variance: float
    cg_iterations: int
    fields: tuple
    seed: int = DEFAULT_SEED
    solver: str = "gmres"

    def __post_init__(self):
        if self.members < 2:
            raise ValueError("the unbiased CRPS needs at least two ensemble members")
        if (self.forecast_frames - 1) % 4:
            raise ValueError("forecast_frames must be one frame plus whole windows of four")


def load_full_settings(value):
    raw = read_yaml(value, "evaluation")
    values = {key: tuple(item) if isinstance(item, list) else item for key, item in raw.items()}
    return FullEvaluationSettings(**values)
