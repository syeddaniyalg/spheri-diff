from dataclasses import dataclass

import torch
from tqdm import tqdm

from spheridiff.data.batches import ordered_batches
from spheridiff.diffusion.sampler import ModelDenoiser

from .metrics import (
    RunningMean,
    crps_unbiased,
    ensemble_variance,
    mean_square_error,
)
from .rollout import rollout

SEED_STRIDE = 7919


@dataclass
class ForecastScores:
    rmse: torch.Tensor
    nrmse: torch.Tensor
    crps: torch.Tensor
    ncrps: torch.Tensor
    ssr: torch.Tensor


class ForecastAccumulators:
    def __init__(self):
        self.physical_error = RunningMean()
        self.normalised_error = RunningMean()
        self.physical_crps = RunningMean()
        self.normalised_crps = RunningMean()
        self.variance = RunningMean()


class ForecastEvaluator:
    headline = ("val_loss/mean", "nrmse/mean", "crps_normalised/mean")

    def __init__(self, dataset, indices, converter, area, process, names, settings, context):
        self.dataset, self.indices = dataset, indices
        self.converter, self.area = converter, area.to(context.device)
        self.process, self.names, self.settings = process, names, settings
        self.device, self.autocast_dtype = context.device, context.autocast_dtype
        self.step_hours = context.step_hours

    def batches(self):
        batches = ordered_batches(self.dataset, self.indices, self.settings.batch_size)
        return tqdm(
            batches,
            total=-(-len(self.indices) // self.settings.batch_size),
            desc="forecast evaluation",
            leave=False,
        )

    def sample_members(self, model, batch, batch_index):
        truth = batch["frames"].to(self.device)
        auxiliary = batch["auxiliary"].to(self.device)
        denoiser = ModelDenoiser(model, self.autocast_dtype)
        members = []
        for member in range(self.settings.members):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.settings.seed + SEED_STRIDE * batch_index + member)
            steps = self.settings.ddim_steps
            members.append(rollout(denoiser, self.process, truth[:, :1], auxiliary, steps, generator))
        return truth, torch.stack(members, dim=1)

    def accumulate(self, accumulators, truth, ensemble):
        physical_truth = self.converter(truth)
        physical_ensemble = self.converter(ensemble)
        mean = ensemble.mean(dim=1)
        physical_mean = physical_ensemble.mean(dim=1)
        accumulators.normalised_error.add(mean_square_error(mean, truth, self.area))
        accumulators.physical_error.add(mean_square_error(physical_mean, physical_truth, self.area))
        accumulators.normalised_crps.add(crps_unbiased(ensemble, truth, self.area))
        accumulators.physical_crps.add(crps_unbiased(physical_ensemble, physical_truth, self.area))
        accumulators.variance.add(ensemble_variance(physical_ensemble, self.area))

    def scores(self, accumulators):
        members = self.settings.members
        spread = accumulators.variance.mean().sqrt()
        error = accumulators.physical_error.mean().sqrt()
        factor = ((members + 1.0) / members) ** 0.5
        return ForecastScores(
            rmse=error,
            nrmse=accumulators.normalised_error.mean().sqrt(),
            crps=accumulators.physical_crps.mean(),
            ncrps=accumulators.normalised_crps.mean(),
            ssr=factor * spread / error.clamp_min(1e-12),
        )

    @torch.no_grad()
    def forecast(self, model):
        accumulators = ForecastAccumulators()
        for index, batch in enumerate(self.batches()):
            truth, ensemble = self.sample_members(model, batch, index)
            self.accumulate(accumulators, truth, ensemble)
        return self.scores(accumulators)

    def evaluate(self, model, include_ssr=False):
        scores = self.forecast(model)
        return flatten_scores(scores, self.names, self.settings.fields, self.step_hours, include_ssr)

    @torch.no_grad()
    def baseline_scores(self, build):
        error = RunningMean()
        normalised = RunningMean()
        for batch in self.batches():
            truth = batch["frames"].to(self.device)
            guess = build(truth)
            normalised.add(mean_square_error(guess, truth, self.area))
            error.add(
                mean_square_error(self.converter(guess), self.converter(truth), self.area)
            )
        return error.mean().sqrt(), normalised.mean().sqrt()

    def baselines(self):
        persistence = lambda truth: truth[:, :1].expand_as(truth)
        channel_mean = lambda truth: torch.zeros_like(truth)
        record = {}
        for name, build in (("persistence", persistence), ("mean", channel_mean)):
            rmse, nrmse = self.baseline_scores(build)
            record.update(flatten_baseline(name, rmse, nrmse, self.names, self.settings, self.step_hours))
        return record


def lead_hours(lead, step_hours):
    return lead * step_hours


def flatten_baseline(name, rmse, nrmse, names, settings, step_hours):
    record = {f"baseline/{name}/nrmse_mean": nrmse[1:].mean().item()}
    for field in settings.fields:
        column = names.index(field)
        for lead in range(1, rmse.shape[0]):
            hours = lead_hours(lead, step_hours)
            record[f"baseline/{name}/rmse/{field}/{hours}h"] = rmse[lead, column].item()
    return record


def flatten_scores(scores, names, fields, step_hours, include_ssr=False):
    record = {
        "nrmse/mean": scores.nrmse[1:].mean().item(),
        "crps_normalised/mean": scores.ncrps[1:].mean().item(),
    }
    for field in fields:
        column = names.index(field)
        for lead in range(1, scores.rmse.shape[0]):
            hours = lead_hours(lead, step_hours)
            record[f"rmse/{field}/{hours}h"] = scores.rmse[lead, column].item()
            record[f"crps/{field}/{hours}h"] = scores.crps[lead, column].item()
            if include_ssr:
                record[f"ssr/{field}/{hours}h"] = scores.ssr[lead, column].item()
    return record
