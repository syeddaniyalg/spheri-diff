import statistics

from .criteria import EDGE_FRACTION, SETTLED_AFTER_STEPS


def edge_means(series, fraction=EDGE_FRACTION):
    count = max(1, int(len(series) * fraction))
    return statistics.fmean(series[:count]), statistics.fmean(series[-count:])


def loss_ratio(step_records):
    losses = [record["loss"] for record in step_records]
    head, tail = edge_means(losses)
    return tail / head


def skipped_fraction(step_records):
    skipped = sum(1 for record in step_records if record["skipped"])
    return skipped / max(1, len(step_records))


def settled_records(step_records):
    return step_records[SETTLED_AFTER_STEPS:] or step_records


def median_samples_per_second(step_records):
    return statistics.median(record["samples_per_second"] for record in settled_records(step_records))


def median_evaluation_seconds(evaluation_records):
    later = [record["evaluation_seconds"] for record in evaluation_records[1:]]
    return statistics.median(later) if later else evaluation_records[0]["evaluation_seconds"]
