import math

import torch
from tqdm import tqdm

from spheridiff.data.batches import ordered_batches
from spheridiff.diffusion.guidance import MMPSDenoiser
from spheridiff.diffusion.operators import (
    FrameMask,
    SparseMask,
    SpatialAverage,
    SpatioTemporalAverage,
    TemporalAverage,
    random_sensor_mask,
    random_space_time_mask,
)
from spheridiff.diffusion.sampler import ModelDenoiser, ddim_sample, native_steps, noised_guess

from .metrics import crps_unbiased, mean_square_error

SEED_STRIDE = 32452843
METHOD_SAMPLERS = {"mmps": "guided", "native": "native", "native_init": "native_init"}


def build_operator(spec, truth, seed):
    frames, height, width = truth.shape[1], truth.shape[-2], truth.shape[-1]
    kind = spec["operator"]
    if kind == "sensor_mask":
        return SparseMask(random_sensor_mask(height, width, spec["fraction"], truth.device, seed))
    if kind == "spatial_average":
        return SpatialAverage(spec["factor"])
    if kind == "frame_mask":
        return FrameMask(frames, spec["observed"], truth.device)
    if kind == "temporal_average":
        return TemporalAverage(frames)
    if kind == "spatiotemporal_average":
        return SpatioTemporalAverage(spec["factor"], frames)
    return SparseMask(random_space_time_mask(frames, height, width, spec["fraction"], truth.device, seed))


def unobserved_error(error_map, mask, area):
    weights = (1.0 - mask).expand_as(error_map) * area
    return (error_map * weights).sum(dim=(1, 2, 3, 4)) / weights.sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)


class ReconstructionScores:
    def __init__(self):
        self.window_error = []
        self.window_unobserved = []
        self.window_consistency = []
        self.window_crps = []
        self.physical = []

    def add(self, values):
        for name, tensor in values.items():
            getattr(self, name).append(tensor.cpu())

    def summarise(self, names, fields):
        error = torch.cat(self.window_error)
        record = {
            "nrmse": math.sqrt(error.mean().item()),
            "nrmse_unobserved": math.sqrt(torch.cat(self.window_unobserved).mean().item()),
            "consistency": math.sqrt(torch.cat(self.window_consistency).mean().item()),
            "ncrps": torch.cat(self.window_crps).mean().item() if self.window_crps else None,
            "window_nmse": error.tolist(),
        }
        physical = torch.cat(self.physical).mean(dim=0).sqrt()
        record["rmse"] = {field: physical[names.index(field)].item() for field in fields}
        return record


class ReconstructionEvaluator:
    def __init__(self, dataset, indices, converter, area, process, names, protocol, device):
        self.dataset, self.indices = dataset, indices
        self.converter, self.area = converter, area.to(device)
        self.process, self.names, self.protocol = process, names, protocol
        self.device = device

    def batches(self, label):
        total = -(-len(self.indices) // self.protocol.batch_size)
        batches = ordered_batches(self.dataset, self.indices, self.protocol.batch_size)
        return tqdm(batches, total=total, desc=label, leave=False)

    def generator(self, batch_index, member):
        generator = torch.Generator(device=self.device)
        return generator.manual_seed(self.protocol.seed + SEED_STRIDE * batch_index + member)

    def guided_member(self, model, operator, observation, auxiliary, channels, generator):
        denoiser = MMPSDenoiser(
            ModelDenoiser(model),
            self.process.schedule,
            operator,
            observation,
            self.protocol.noise_variance,
            self.protocol.solver_iterations,
            self.protocol.solver,
            self.process.noise.covariance(self.device),
        )
        steps = self.protocol.ddim_steps
        return ddim_sample(denoiser, self.process, auxiliary, channels, steps, generator=generator)

    def native_member(self, model, operator, observation, auxiliary, channels, generator):
        steps = self.protocol.ddim_steps
        return ddim_sample(ModelDenoiser(model), self.process, auxiliary, channels, steps, generator=generator)

    def native_init_member(self, model, operator, observation, auxiliary, channels, generator):
        level = self.protocol.native_start_level
        initial = noised_guess(self.process, operator.baseline(observation), level, generator)
        steps = native_steps(self.protocol.ddim_steps, level)
        return ddim_sample(
            ModelDenoiser(model),
            self.process,
            auxiliary,
            channels,
            steps,
            generator=generator,
            start_level=level,
            initial=initial,
        )

    def sample(self, method, model, operator, observation, auxiliary, channels, batch_index):
        members = []
        for member in range(self.protocol.members):
            generator = self.generator(batch_index, member)
            sampler = getattr(self, f"{METHOD_SAMPLERS[method]}_member")
            members.append(sampler(model, operator, observation, auxiliary, channels, generator))
        return torch.stack(members, dim=1)

    def batch_scores(self, ensemble, truth, operator, observation):
        mean = ensemble.mean(dim=1)
        error_map = (mean - truth) ** 2
        mask = getattr(operator, "mask", None)
        window_error = mean_square_error(mean, truth, self.area).mean(dim=(1, 2))
        unobserved = window_error if mask is None else unobserved_error(error_map, mask, self.area)
        values = {
            "window_error": window_error,
            "window_unobserved": unobserved,
            "window_consistency": ((operator.apply(mean) - observation) ** 2).flatten(1).mean(dim=1),
            "physical": mean_square_error(self.converter(mean), self.converter(truth), self.area).mean(dim=1),
        }
        if ensemble.shape[1] > 1:
            values["window_crps"] = crps_unbiased(ensemble, truth, self.area).mean(dim=(1, 2))
        return values

    def run(self, model, task, spec, method):
        scores = ReconstructionScores()
        for index, batch in enumerate(self.batches(f"{task} {method}")):
            truth = batch["frames"].to(self.device)
            auxiliary = batch["auxiliary"].to(self.device)
            operator = build_operator(spec, truth, self.protocol.seed)
            observation = operator.apply(truth)
            if method == "baseline":
                ensemble = operator.baseline(observation)[:, None]
            else:
                ensemble = self.sample(method, model, operator, observation, auxiliary, truth.shape[2], index)
            scores.add(self.batch_scores(ensemble, truth, operator, observation))
        return scores.summarise(self.names, self.protocol.fields)

    def evaluate(self, model):
        results = {}
        for task, spec in self.protocol.tasks.items():
            results[task] = {"baseline": self.run(model, task, spec, "baseline")}
            for method in self.protocol.methods:
                results[task][method] = self.run(model, task, spec, method)
        return results
