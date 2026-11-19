from dataclasses import dataclass

from spheridiff.common.config import read_yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    noise: str
    projection: str
    architecture: dict


def load_model_spec(value):
    raw = read_yaml(value, "model")
    diffusion = raw["diffusion"]
    return ModelSpec(raw["name"], diffusion["noise"], diffusion["projection"], raw["architecture"])
