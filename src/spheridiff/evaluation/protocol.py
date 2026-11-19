import hashlib
import json
from dataclasses import dataclass

from spheridiff.common.config import read_yaml
from spheridiff.diffusion.solvers import SOLVERS

OPERATORS = (
    "sensor_mask",
    "spatial_average",
    "frame_mask",
    "temporal_average",
    "spatiotemporal_average",
    "space_time_mask",
)
METHODS = ("mmps", "native", "native_init")


@dataclass(frozen=True)
class FinalProtocol:
    name: str
    split: str
    windows: int
    members: int
    ddim_steps: int
    batch_size: int
    forecast_frames: int
    seed: int
    precision: str
    fields: tuple
    noise_variance: float
    solver_iterations: int
    solver: str
    native_start_level: float
    methods: tuple
    tasks: dict
    digest: str

    def __post_init__(self):
        if self.members < 2:
            raise ValueError("the unbiased CRPS needs at least two ensemble members")
        if (self.forecast_frames - 1) % 4:
            raise ValueError("forecast_frames must be one frame plus whole windows of four")
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown solver '{self.solver}'")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}")
        for name, spec in self.tasks.items():
            if spec["operator"] not in OPERATORS:
                raise ValueError(f"task {name} uses an unknown operator '{spec['operator']}'")

    def summary(self):
        return {"name": self.name, "digest": self.digest, "split": self.split, "windows": self.windows}


def protocol_digest(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()


def protocol_from_dict(raw):
    return FinalProtocol(
        name=raw["name"],
        split=raw["split"],
        windows=raw["windows"],
        members=raw["members"],
        ddim_steps=raw["ddim_steps"],
        batch_size=raw["batch_size"],
        forecast_frames=raw["forecast_frames"],
        seed=raw["seed"],
        precision=raw["precision"],
        fields=tuple(raw["fields"]),
        noise_variance=raw["guidance"]["noise_variance"],
        solver_iterations=raw["guidance"]["iterations"],
        solver=raw["guidance"]["solver"],
        native_start_level=raw["native_init"]["start_level"],
        methods=tuple(raw["methods"]),
        tasks=raw["tasks"],
        digest=protocol_digest(raw),
    )


def load_protocol(value):
    return protocol_from_dict(read_yaml(value, "protocol"))
