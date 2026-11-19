import torch

from spheridiff.diffusion.sampler import ddim_sample

WINDOW = 5


def window_context(current, auxiliary_window):
    batch, frames, _, height, width = auxiliary_window.shape
    context = torch.zeros(batch, frames, current.shape[2], height, width, device=current.device)
    context[:, :1] = current
    mask = torch.zeros_like(context, dtype=torch.bool)
    mask[:, :1] = True
    return context, mask


def rollout(denoiser, process, first_frame, auxiliary, steps, generator=None):
    horizon = auxiliary.shape[1]
    if (horizon - 1) % (WINDOW - 1):
        raise ValueError("the horizon must be one frame plus whole windows of four new frames")
    predictions, current = [first_frame], first_frame
    for start in range(0, horizon - 1, WINDOW - 1):
        window = auxiliary[:, start : start + WINDOW]
        context, mask = window_context(current, window)
        output = ddim_sample(
            denoiser, process, window, first_frame.shape[2], steps, context, mask, generator
        )
        predictions.append(output[:, 1:])
        current = output[:, -1:]
    return torch.cat(predictions, dim=1)
