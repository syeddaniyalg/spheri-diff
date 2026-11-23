import torch


def inform_levels(model, levels):
    setter = getattr(model, "set_frame_levels", None)
    if setter is not None:
        setter(levels)


def denoising_loss(model, process, weights, clean, auxiliary, generator=None, levels=None):
    if levels is None:
        batch, frames = clean.shape[:2]
        levels = process.sample_levels(batch, frames, clean.device, generator)
    noisy, target = process.corrupt(clean, levels, generator)
    inform_levels(model, levels)
    prediction = model(noisy, auxiliary)
    return ((prediction.float() - target) ** 2 * weights).mean()


def fixed_level_loss(model, process, weights, clean, auxiliary, level, generator=None):
    levels = torch.full(clean.shape[:2], level, device=clean.device)
    return denoising_loss(model, process, weights, clean, auxiliary, generator, levels)
