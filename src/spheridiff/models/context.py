from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelContext:
    channels: int
    auxiliary_channels: int
    frames: int
    grid: object
    sphere: dict = field(default_factory=dict)
