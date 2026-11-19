import yaml

from helpers import make_cache, make_config
from spheridiff.data.statistics import build_statistics

TINY_WIND = {
    "name": "wind",
    "diffusion": {"noise": "gaussian", "projection": "identity"},
    "architecture": {
        "widths": [32, 64, 64, 64],
        "block_kinds": ["res", "res", "transformer", "transformer"],
        "blocks_per_level": [1, 1, 1],
        "middle_blocks": 1,
        "embedding_dim": 16,
        "heads": 4,
        "patch_size": 1,
        "norm_groups": 8,
    },
}

TINY_SPHERIDIFF = {
    "name": "spheridiff",
    "diffusion": {"noise": "spectral", "projection": "band_limit"},
    "architecture": {"widths": [16, 32, 32, 16], "heads": 4},
}

EVALUATION_FIELDS = ["500_temperature", "total_precipitation_6hr"]


def data_dict(cache_root):
    return {
        "source": {"url": "unused", "storage_options": {}, "stored_levels": 13},
        "grid": {"height": 8, "width": 16},
        "window": {"frames": 5, "step_hours": 6},
        "transforms": {"precipitation_channels": ["total_precipitation_6hr"]},
        "splits": {"train": [2010, 2010], "val": [2011, 2011]},
        "channels": {
            "pressure_variables": ["temperature"],
            "levels": [500, 850],
            "surface_variables": ["total_precipitation_6hr"],
            "static_fields": ["land_sea_mask", "soil_type", "geopotential_at_surface"],
        },
        "cache": {"root": str(cache_root)},
    }


def train_dict(
    steps, batch_size=4, evaluate_every=0, fixed_windows=0, learning_rate=3e-3, checkpoint_every=0
):
    return {
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "min_learning_rate": 1e-5,
        "warmup_fraction": 0.1,
        "weight_decay": 0.0,
        "gradient_clip": 0.8,
        "ema_decay": 0.99,
        "precision": "fp16",
        "evaluate_every": evaluate_every,
        "checkpoint_every": checkpoint_every,
        "train_split": "train",
        "fixed_windows": fixed_windows,
        "source_levels": [50, 500, 850, 1000],
        "evaluation": {
            "windows": 4,
            "members": 2,
            "ddim_steps": 2,
            "batch_size": 2,
            "levels": [0.2, 0.8],
            "fields": EVALUATION_FIELDS,
        },
    }


def full_evaluation_dict():
    return {
        "split": "val",
        "windows": 2,
        "members": 2,
        "ddim_steps": 2,
        "batch_size": 2,
        "forecast_frames": 9,
        "spatial_factor": 4,
        "sparse_fraction": 0.1,
        "noise_variance": 0.0015,
        "cg_iterations": 1,
        "fields": EVALUATION_FIELDS,
    }


def dump(path, content):
    path.write_text(yaml.safe_dump(content))
    return str(path)


def write_experiment(directory, model, steps=20, **train_options):
    cache_root = directory / "cache"
    config = make_config(cache_root)
    make_cache(config)
    build_statistics(config)
    spec = {
        "name": "tiny",
        "data": dump(directory / "data.yaml", data_dict(cache_root)),
        "model": dump(directory / "model.yaml", model),
        "train": dump(directory / "train.yaml", train_dict(steps, **train_options)),
        "evaluation": dump(directory / "evaluation.yaml", full_evaluation_dict()),
        "seed": 0,
    }
    return dump(directory / "experiment.yaml", spec), cache_root
