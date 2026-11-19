import json

import torch

from spheridiff.data.batches import evenly_spaced_indices
from spheridiff.data.loading import build_dataset
from spheridiff.evaluation.context import EvaluationContext
from spheridiff.evaluation.final import FinalEvaluation
from spheridiff.evaluation.forecast import ForecastEvaluator
from spheridiff.evaluation.reconstruction import ReconstructionEvaluator
from spheridiff.models.registry import count_parameters
from spheridiff.training.precision import autocast_dtype
from spheridiff.training.runlog import RunPaths

from .build import area_tensor, build_components, converter_for
from .spec import load_experiment

RESULT_NAME = "final_evaluation.json"


def final_datasets(config, protocol):
    return {
        "forecast": build_dataset(config, protocol.split, window_frames=protocol.forecast_frames),
        "tasks": build_dataset(config, protocol.split),
    }


def build_final_evaluation(components, protocol, device, datasets):
    context = EvaluationContext(device, autocast_dtype(protocol.precision), components.config.step_hours)
    forecast_indices = evenly_spaced_indices(len(datasets["forecast"]), protocol.windows)
    task_indices = evenly_spaced_indices(len(datasets["tasks"]), protocol.windows)
    shared = (converter_for(components), area_tensor(components), components.process, components.names)
    forecast = ForecastEvaluator(datasets["forecast"], forecast_indices, *shared, protocol, context)
    reconstruction = ReconstructionEvaluator(datasets["tasks"], task_indices, *shared, protocol, device)
    return FinalEvaluation(protocol, forecast, reconstruction, forecast_indices, task_indices)


def describe_run(spec, components, paths, state):
    return {
        "experiment": spec.name,
        "model": components.model_spec.name,
        "architecture": components.model_spec.architecture,
        "parameters": count_parameters(components.model),
        "trained_steps": state["step"],
        "training_complete": paths.final.exists(),
        "seed": spec.seed,
    }


def run_final_evaluation(experiment, protocol, runs_root, cache_root, device, datasets=None):
    spec = load_experiment(experiment)
    components = build_components(spec, cache_root)
    paths = RunPaths(runs_root, spec.name)
    state = torch.load(paths.weights, map_location=device, weights_only=False)
    components.model.load_state_dict(state["model"])
    components.model.to(device)
    datasets = datasets or final_datasets(components.config, protocol)
    evaluation = build_final_evaluation(components, protocol, device, datasets)
    result = evaluation.run(components.model)
    result["run"] = describe_run(spec, components, paths, state)
    (paths.root / RESULT_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
