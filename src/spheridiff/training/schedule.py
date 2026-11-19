import math


def learning_rate_at(step, total_steps, peak, minimum, warmup_steps):
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))
