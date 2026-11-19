import dataclasses
from dataclasses import dataclass

import torch

from spheridiff.common.config import load_data_config
from spheridiff.data.auxiliary import auxiliary_channel_count
from spheridiff.data.batches import IndexedDataset, evenly_spaced_indices
from spheridiff.data.channels import channel_names
from spheridiff.data.loading import build_dataset, load_dataset_statistics, load_grid
from spheridiff.data.postprocess import PhysicalConverter
from spheridiff.diffusion.process import build_process
from spheridiff.diffusion.weights import loss_weights
from spheridiff.evaluation.context import EvaluationContext
from spheridiff.evaluation.forecast import ForecastEvaluator
from spheridiff.evaluation.full import FullEvaluation
from spheridiff.evaluation.periodic import PeriodicEvaluation
from spheridiff.evaluation.settings import load_full_settings
from spheridiff.evaluation.tasks import TaskEvaluator
from spheridiff.evaluation.validation import ValidationLoss
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spec import load_model_spec
from spheridiff.training.loop import Trainer, TrainingParts
from spheridiff.training.precision import autocast_dtype, resolve_precision
from spheridiff.training.runlog import RunPaths
from spheridiff.training.settings import load_train_settings

from .spec import load_experiment


@dataclass
class Components:
    spec: object
    config: object
    grid: object
    statistics: object
    names: list
    process: object
    model_spec: object
    model: object


def build_components(spec, cache_root=None):
    config = load_data_config(spec.data, cache_root)
    grid, statistics = load_grid(config), load_dataset_statistics(config)
    names = channel_names(config.channels)
    model_spec = load_model_spec(spec.model)
    process = build_process(model_spec.noise, model_spec.projection, grid)
    auxiliary = auxiliary_channel_count(config.channels.static_fields)
    context = ModelContext(len(names), auxiliary, config.frames, grid)
    torch.manual_seed(spec.seed)
    model = build_model(model_spec.name, model_spec.architecture, context)
    return Components(spec, config, grid, statistics, names, process, model_spec, model)


def converter_for(components):
    config = components.config
    return PhysicalConverter(components.statistics, components.names, config.precipitation_channels)


def area_tensor(components):
    return torch.as_tensor(components.grid.area_weights, dtype=torch.float32)


def training_dataset(components, settings):
    dataset = build_dataset(components.config, settings.train_split, show_progress=True)
    if settings.fixed_windows > 0:
        return IndexedDataset(dataset, range(settings.fixed_windows))
    return dataset


def evaluation_dataset(components, settings, training):
    if settings.fixed_windows > 0:
        return training
    return build_dataset(components.config, "val", show_progress=True)


def periodic_evaluation(components, settings, dataset, device):
    evaluation = settings.evaluation
    indices = evenly_spaced_indices(len(dataset), evaluation.windows)
    dtype = autocast_dtype(resolve_precision(settings.precision, device))
    context = EvaluationContext(device, dtype, components.config.step_hours)
    weights = loss_weights(components.grid, components.names, components.statistics, settings.source_levels)
    validation = ValidationLoss(dataset, indices, components.process, weights, evaluation, context)
    forecast = ForecastEvaluator(
        dataset,
        indices,
        converter_for(components),
        area_tensor(components),
        components.process,
        components.names,
        evaluation,
        context,
    )
    return PeriodicEvaluation(validation, forecast)


def run_metadata(spec, settings, components, device):
    model_spec = components.model_spec
    return {
        "experiment": dataclasses.asdict(spec),
        "train": dataclasses.asdict(settings),
        "model": dataclasses.asdict(model_spec),
        "parameters": count_parameters(components.model),
        "data": {"cache_root": str(components.config.cache_root), "splits": components.config.splits},
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def build_trainer(spec, device, runs_root, cache_root=None, settings=None, run_name=None):
    components = build_components(spec, cache_root)
    settings = settings or load_train_settings(spec.train)
    dataset = training_dataset(components, settings)
    evaluator = periodic_evaluation(components, settings, evaluation_dataset(components, settings, dataset), device)
    weights = loss_weights(components.grid, components.names, components.statistics, settings.source_levels)
    metadata = run_metadata(spec, settings, components, device)
    parts = TrainingParts(components.model, components.process, weights, dataset, evaluator, metadata)
    paths = RunPaths(runs_root, run_name or spec.name)
    return Trainer(parts, settings, paths, device, spec.seed, run_name or spec.name), components


def build_full_evaluation(components, settings, device):
    config = components.config
    context = EvaluationContext(device, None, config.step_hours)
    forecast_data = build_dataset(config, settings.split, window_frames=settings.forecast_frames)
    task_data = build_dataset(config, settings.split)
    forecast = ForecastEvaluator(
        forecast_data,
        evenly_spaced_indices(len(forecast_data), settings.windows),
        converter_for(components),
        area_tensor(components),
        components.process,
        components.names,
        settings,
        context,
    )
    tasks = TaskEvaluator(
        task_data,
        evenly_spaced_indices(len(task_data), settings.windows),
        converter_for(components),
        area_tensor(components),
        components.process,
        components.names,
        settings,
        context,
    )
    return FullEvaluation(forecast, tasks)


def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device)


def prepare_experiment(name):
    spec = load_experiment(name)
    return spec, load_train_settings(spec.train), load_full_settings(spec.evaluation)
