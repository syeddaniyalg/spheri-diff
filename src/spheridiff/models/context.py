from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContext:
    channels: int
    auxiliary_channels: int
    frames: int
    grid: object
