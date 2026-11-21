from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


@dataclass(frozen=True)
class SourceConfig:
    url: str
    storage_options: dict
    stored_levels: int


@dataclass(frozen=True)
class GridConfig:
    height: int
    width: int
    latitude_range: tuple = None
    longitude_range: tuple = None

    @property
    def regional(self):
        return self.latitude_range is not None


@dataclass(frozen=True)
class ChannelConfig:
    pressure_variables: tuple
    levels: tuple
    surface_variables: tuple
    static_fields: tuple


@dataclass(frozen=True)
class DataConfig:
    source: SourceConfig
    grid: GridConfig
    channels: ChannelConfig
    frames: int
    step_hours: int
    precipitation_channels: tuple
    splits: dict
    cache_root: Path

    @property
    def step_seconds(self):
        return self.step_hours * SECONDS_PER_HOUR


def resolve_config_path(value, folder="data"):
    path = Path(value)
    if path.is_file():
        return path
    named = CONFIG_ROOT / folder / f"{value}.yaml"
    if named.is_file():
        return named
    raise FileNotFoundError(f"no {folder} config found for '{value}'")


def read_yaml(value, folder):
    with open(resolve_config_path(value, folder)) as handle:
        return yaml.safe_load(handle)


def resolve_cache_root(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def grid_config(raw):
    converted = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
    return GridConfig(**converted)


def load_data_config(path, cache_root=None):
    raw = read_yaml(path, "data")
    channels = {key: tuple(value) for key, value in raw["channels"].items()}
    return DataConfig(
        source=SourceConfig(**raw["source"]),
        grid=grid_config(raw["grid"]),
        channels=ChannelConfig(**channels),
        frames=raw["window"]["frames"],
        step_hours=raw["window"]["step_hours"],
        precipitation_channels=tuple(raw["transforms"]["precipitation_channels"]),
        splits={name: tuple(years) for name, years in raw["splits"].items()},
        cache_root=resolve_cache_root(cache_root or raw["cache"]["root"]),
    )
