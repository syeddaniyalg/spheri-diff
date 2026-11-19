import dataclasses
import math

import torch

from spheridiff.data.auxiliary import auxiliary_channel_count
from spheridiff.evaluation.settings import TASKS, load_full_settings
from spheridiff.experiments.build import build_components, build_full_evaluation, build_trainer
from spheridiff.models.registry import count_parameters
from spheridiff.training.runlog import RunPaths, read_records
from spheridiff.training.settings import load_train_settings

from .criteria import (
    BASELINE_KEY,
    HEADLINE_KEY,
    MAX_SKIPPED_FRACTION,
    OVERFIT_LOSS_RATIO,
    PERSISTENCE_KEY,
    RESUME_GAP_RATIO,
    SHORT_LOSS_RATIO,
)
from .curves import loss_ratio, skipped_fraction
from .estimate import estimate_pilot_time

RESUME_STEPS = 6
RESUME_SPLIT = 3


def static_stage(spec, device, runs_root, cache_root):
    components = build_components(spec, cache_root)
    model = components.model.to(device)
    config = components.config
    channels = len(components.names)
    height, width = config.grid.height, config.grid.width
    auxiliary = auxiliary_channel_count(config.channels.static_fields)
    noisy = torch.randn(2, config.frames, channels, height, width, device=device)
    output = model(noisy, torch.randn(2, config.frames, auxiliary, height, width, device=device))
    zero_start = torch.count_nonzero(output).item() == 0
    return {
        "passed": zero_start and output.shape == noisy.shape,
        "parameters": count_parameters(model),
        "output_shape": list(output.shape),
    }


def overfit_stage(spec, device, runs_root, cache_root):
    settings = load_train_settings("overfit")
    trainer, _ = build_trainer(spec, device, runs_root, cache_root, settings, f"{spec.name}_overfit")
    trainer.run(resume=False)
    steps = read_records(trainer.paths.steps)
    evaluations = read_records(trainer.paths.evaluations)
    ratio = loss_ratio(steps)
    final, baseline = evaluations[-1][HEADLINE_KEY], evaluations[0][BASELINE_KEY]
    return {
        "passed": ratio < OVERFIT_LOSS_RATIO and final < baseline,
        "loss_ratio": ratio,
        "final_nrmse": final,
        "mean_baseline_nrmse": baseline,
    }


def short_stage(spec, device, runs_root, cache_root):
    trainer, _ = build_trainer(spec, device, runs_root, cache_root)
    trainer.run(resume=False)
    steps = read_records(trainer.paths.steps)
    evaluations = read_records(trainer.paths.evaluations)
    ratio, skipped = loss_ratio(steps), skipped_fraction(steps)
    final, baseline = evaluations[-1][HEADLINE_KEY], evaluations[0][BASELINE_KEY]
    return {
        "passed": ratio < SHORT_LOSS_RATIO and final < baseline and skipped < MAX_SKIPPED_FRACTION,
        "loss_ratio": ratio,
        "skipped_fraction": skipped,
        "final_nrmse": final,
        "mean_baseline_nrmse": baseline,
        "persistence_nrmse": evaluations[0][PERSISTENCE_KEY],
    }


def parameter_vector(model):
    return torch.cat([parameter.detach().flatten().float().cpu() for parameter in model.parameters()])


def resume_stage(spec, device, runs_root, cache_root):
    base = load_train_settings("overfit")
    settings = dataclasses.replace(base, steps=RESUME_STEPS, evaluate_every=0, checkpoint_every=RESUME_SPLIT)
    continuous, _ = build_trainer(spec, device, runs_root, cache_root, settings, f"{spec.name}_resume_a")
    start = parameter_vector(continuous.model)
    continuous.run(resume=False)
    first, _ = build_trainer(spec, device, runs_root, cache_root, settings, f"{spec.name}_resume_b")
    first.stop_at = RESUME_SPLIT
    first.run(resume=False)
    second, _ = build_trainer(spec, device, runs_root, cache_root, settings, f"{spec.name}_resume_b")
    second.run(resume=True)
    gap = (parameter_vector(second.model) - parameter_vector(continuous.model)).norm()
    scale = (parameter_vector(continuous.model) - start).norm()
    ratio = (gap / scale).item()
    return {"passed": second.step == RESUME_STEPS and ratio < RESUME_GAP_RATIO, "gap_ratio": ratio}


def tasks_stage(spec, device, runs_root, cache_root):
    components = build_components(spec, cache_root)
    settings = load_full_settings(spec.evaluation)
    evaluation = build_full_evaluation(components, settings, device)
    record = evaluation.run(components.model.to(device))
    covered = {key.split("/")[0] for key in record}
    finite = all(math.isfinite(value) for value in record.values())
    return {"passed": finite and set(TASKS) <= covered, "values": len(record)}


def timing_stage(spec, device, runs_root, cache_root):
    paths = RunPaths(runs_root, spec.name)
    steps, evaluations = read_records(paths.steps), read_records(paths.evaluations)
    if not steps or not evaluations:
        return {"passed": False, "reason": "the short run has not been recorded"}
    smoke_settings, pilot_settings = load_train_settings(spec.train), load_train_settings(spec.pilot_train)
    estimate = estimate_pilot_time(steps, evaluations, smoke_settings, pilot_settings)
    return {"passed": True, **estimate}


STAGES = {
    "static": static_stage,
    "overfit": overfit_stage,
    "short": short_stage,
    "resume": resume_stage,
    "tasks": tasks_stage,
    "timing": timing_stage,
}
