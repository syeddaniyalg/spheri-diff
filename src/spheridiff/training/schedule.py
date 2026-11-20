import math


def learning_rate_at(step, total_steps, peak, minimum, warmup_steps):
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def cooldown_rate_at(step, start_step, total_steps, peak, minimum):
    span = max(1, total_steps - start_step)
    progress = min(1.0, max(0.0, (step - start_step) / span))
    return minimum + (peak - minimum) * (1.0 - math.sqrt(progress))


def scheduled_rate(settings, step):
    if settings.cooldown_start > 0:
        return cooldown_rate_at(
            step, settings.cooldown_start, settings.steps, settings.cooldown_peak, settings.min_learning_rate
        )
    return learning_rate_at(
        step, settings.steps, settings.learning_rate, settings.min_learning_rate, settings.warmup_steps
    )
