import torch


class ModelDenoiser:
    def __init__(self, model, autocast_dtype=None):
        self.model = model
        self.autocast_dtype = autocast_dtype

    def __call__(self, noisy, auxiliary, level):
        device_type = noisy.device.type
        enabled = self.autocast_dtype is not None
        with torch.autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=enabled):
            return self.model(noisy, auxiliary).float()


def initial_state(process, shape, device, generator):
    alpha, sigma = process.schedule(1.0)
    scale = (alpha**2 + sigma**2) ** 0.5
    return scale * process.noise.sample(shape, device, generator)


def ddim_step(denoiser, process, state, auxiliary, levels, context, mask):
    level, next_level = levels
    alpha, sigma = process.schedule(level)
    next_alpha, next_sigma = process.schedule(next_level)
    visible = state if mask is None else torch.where(mask, context, state)
    estimate = denoiser(visible, auxiliary, level)
    direction = (state - alpha * estimate) / sigma
    return next_alpha * estimate + next_sigma * direction


def native_steps(steps, start_level):
    return max(2, round(steps * start_level))


def noised_guess(process, guess, level, generator):
    levels = torch.full(guess.shape[:2], level, device=guess.device)
    return process.corrupt(guess, levels, generator)[0]


@torch.no_grad()
def ddim_sample(
    denoiser,
    process,
    auxiliary,
    channels,
    steps,
    context=None,
    mask=None,
    generator=None,
    start_level=1.0,
    initial=None,
):
    batch, frames, _, height, width = auxiliary.shape
    shape = (batch, frames, channels, height, width)
    state = initial_state(process, shape, auxiliary.device, generator) if initial is None else initial
    grid = torch.linspace(start_level, 0.0, steps + 1).tolist()
    for levels in zip(grid[:-1], grid[1:]):
        state = ddim_step(denoiser, process, state, auxiliary, levels, context, mask)
    return state if mask is None else torch.where(mask, context, state)
