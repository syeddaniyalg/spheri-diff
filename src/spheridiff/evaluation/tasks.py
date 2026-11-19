import torch
from tqdm import tqdm

from spheridiff.data.batches import ordered_batches
from spheridiff.diffusion.guidance import MMPSDenoiser
from spheridiff.diffusion.operators import (
    SparseMask,
    SpatialAverage,
    TemporalAverage,
    random_sensor_mask,
)
from spheridiff.diffusion.sampler import ModelDenoiser, ddim_sample

from .metrics import RunningMean, mean_square_error

SEED_STRIDE = 15485863


def build_operator(task, truth, settings):
    if task == "spatial_downscaling":
        return SpatialAverage(settings.spatial_factor)
    if task == "temporal_downscaling":
        return TemporalAverage(truth.shape[1])
    if task == "sparse_reconstruction":
        height, width = truth.shape[-2:]
        mask = random_sensor_mask(height, width, settings.sparse_fraction, truth.device, settings.seed)
        return SparseMask(mask)
    raise ValueError(f"unknown task '{task}'")


class TaskEvaluator:
    def __init__(self, dataset, indices, converter, area, process, names, settings, context):
        self.dataset, self.indices = dataset, indices
        self.converter, self.area = converter, area.to(context.device)
        self.process, self.names, self.settings = process, names, settings
        self.device = context.device

    def batches(self, task):
        batches = ordered_batches(self.dataset, self.indices, self.settings.batch_size)
        total = -(-len(self.indices) // self.settings.batch_size)
        return tqdm(batches, total=total, desc=task, leave=False)

    def reconstruct(self, model, truth, auxiliary, operator, observation, batch_index):
        members = []
        for member in range(self.settings.members):
            denoiser = MMPSDenoiser(
                ModelDenoiser(model),
                self.process.schedule,
                operator,
                observation,
                self.settings.noise_variance,
                self.settings.cg_iterations,
                self.settings.solver,
                self.process.noise.covariance(self.device),
            )
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.settings.seed + SEED_STRIDE * batch_index + member)
            steps = self.settings.ddim_steps
            members.append(ddim_sample(denoiser, self.process, auxiliary, truth.shape[2], steps, generator=generator))
        return torch.stack(members, dim=0).mean(dim=0)

    def add_errors(self, accumulators, guess, truth):
        accumulators[0].add(mean_square_error(guess, truth, self.area))
        physical = mean_square_error(self.converter(guess), self.converter(truth), self.area)
        accumulators[1].add(physical)

    def run_task(self, model, task):
        model_scores, baseline_scores = (RunningMean(), RunningMean()), (RunningMean(), RunningMean())
        for index, batch in enumerate(self.batches(task)):
            truth = batch["frames"].to(self.device)
            auxiliary = batch["auxiliary"].to(self.device)
            operator = build_operator(task, truth, self.settings)
            observation = operator.apply(truth)
            guess = self.reconstruct(model, truth, auxiliary, operator, observation, index)
            self.add_errors(model_scores, guess, truth)
            self.add_errors(baseline_scores, operator.baseline(observation), truth)
        return model_scores, baseline_scores

    def summarise(self, prefix, scores):
        normalised = scores[0].mean().mean(dim=0).sqrt()
        physical = scores[1].mean().mean(dim=0).sqrt()
        record = {f"{prefix}/nrmse_mean": normalised.mean().item()}
        for field in self.settings.fields:
            record[f"{prefix}/rmse/{field}"] = physical[self.names.index(field)].item()
        return record

    def evaluate(self, model, task):
        model_scores, baseline_scores = self.run_task(model, task)
        record = self.summarise(task, model_scores)
        record.update(self.summarise(f"{task}/baseline", baseline_scores))
        return record
