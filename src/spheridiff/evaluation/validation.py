import torch

from spheridiff.data.batches import ordered_batches
from spheridiff.diffusion.objective import fixed_level_loss

SEED_STRIDE = 104729


class ValidationLoss:
    def __init__(self, dataset, indices, process, weights, settings, context):
        self.dataset, self.indices = dataset, indices
        self.process, self.settings = process, settings
        self.device, self.autocast_dtype = context.device, context.autocast_dtype
        self.weights = weights.to(context.device)

    def level_loss(self, model, batch, level, batch_index):
        clean = batch["frames"].to(self.device)
        auxiliary = batch["auxiliary"].to(self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.settings.seed + SEED_STRIDE * batch_index)
        enabled = self.autocast_dtype is not None
        with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=enabled):
            loss = fixed_level_loss(model, self.process, self.weights, clean, auxiliary, level, generator)
        return loss.item() * clean.shape[0]

    @torch.no_grad()
    def evaluate(self, model):
        totals = {level: 0.0 for level in self.settings.levels}
        for index, batch in enumerate(ordered_batches(self.dataset, self.indices, self.settings.batch_size)):
            for level in totals:
                totals[level] += self.level_loss(model, batch, level, index)
        record = {f"val_loss/{level}": total / len(self.indices) for level, total in totals.items()}
        record["val_loss/mean"] = sum(record.values()) / len(record)
        return record
