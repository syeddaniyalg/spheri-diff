from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvaluationContext:
    device: torch.device
    autocast_dtype: object
    step_hours: int
