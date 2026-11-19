from .criteria import TIMING_MARGIN
from .curves import median_evaluation_seconds, median_samples_per_second


def evaluation_cost(settings):
    evaluation = settings.evaluation
    return evaluation.windows * evaluation.members * evaluation.ddim_steps


def estimate_pilot_time(step_records, evaluation_records, smoke_settings, pilot_settings):
    samples_per_second = median_samples_per_second(step_records)
    training_seconds = pilot_settings.steps * pilot_settings.batch_size / samples_per_second
    scale = evaluation_cost(pilot_settings) / evaluation_cost(smoke_settings)
    per_evaluation = median_evaluation_seconds(evaluation_records) * scale
    evaluations = pilot_settings.steps // pilot_settings.evaluate_every + 1
    total = training_seconds + evaluations * per_evaluation
    return {
        "samples_per_second": samples_per_second,
        "pilot_steps": pilot_settings.steps,
        "training_hours": training_seconds / 3600,
        "evaluation_hours": evaluations * per_evaluation / 3600,
        "total_hours": total / 3600,
        "total_hours_with_margin": total * TIMING_MARGIN / 3600,
    }
