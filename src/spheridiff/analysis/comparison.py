import json
from pathlib import Path

import numpy as np

from spheridiff.common.seed import DEFAULT_SEED
from spheridiff.training.runlog import RunPaths, read_records

HEADLINE = "nrmse/mean"
FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
SUSTAIN = 2
RESAMPLES = 2000
CONFIDENCE = 0.95


def evaluations_of(runs_root, name, key=HEADLINE):
    records = read_records(RunPaths(runs_root, name).evaluations)
    return [record for record in records if key in record and record["step"] > 0]


def first_sustained(values, level, count=SUSTAIN):
    for index in range(len(values)):
        if all(value <= level for value in values[index : index + count]):
            return index
    return None


def crossing_record(records, key, level):
    index = first_sustained([record[key] for record in records], level)
    if index is None:
        return None
    record = records[index]
    return {name: record[name] for name in ("step", "samples_seen", "train_seconds")}


def nearest_record(records, step):
    return min(records, key=lambda record: abs(record["step"] - step))


def fraction_table(records, key, fractions=FRACTIONS):
    last = records[-1]["step"]
    table = {}
    for fraction in fractions:
        record = nearest_record(records, fraction * last)
        table[str(fraction)] = {"step": record["step"], "value": record[key]}
    return table


def curve_area(records, key):
    steps = np.array([record["step"] for record in records], dtype=float)
    values = np.array([record[key] for record in records], dtype=float)
    if len(steps) < 2:
        return float(values[0])
    pieces = (values[1:] + values[:-1]) / 2.0 * np.diff(steps)
    return float(pieces.sum() / (steps[-1] - steps[0]))


def window_scores(runs_root, name):
    records = read_records(RunPaths(runs_root, name).windows)
    return {record["step"]: record for record in records}


def compared_steps(run_windows, reference_windows, at_final):
    if at_final:
        return (max(run_windows), max(reference_windows)) if run_windows and reference_windows else None
    shared = sorted(set(run_windows) & set(reference_windows))
    return (shared[-1], shared[-1]) if shared else None


def bootstrap_interval(run_errors, reference_errors, seed):
    count = len(run_errors)
    choices = np.random.default_rng(seed).integers(0, count, (RESAMPLES, count))
    resampled = np.sqrt(run_errors[choices].mean(axis=1)) - np.sqrt(reference_errors[choices].mean(axis=1))
    edge = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(resampled, [edge, 1.0 - edge])
    return float(low), float(high), float((resampled < 0).mean())


def paired_bootstrap(runs_root, run, reference, seed=DEFAULT_SEED, at_final=False, reference_root=None):
    run_windows = window_scores(runs_root, run)
    reference_windows = window_scores(reference_root or runs_root, reference)
    steps = compared_steps(run_windows, reference_windows, at_final)
    if steps is None:
        return None
    first, second = run_windows[steps[0]], reference_windows[steps[1]]
    if first["indices"] != second["indices"]:
        return None
    run_errors, reference_errors = np.array(first["nmse"]), np.array(second["nmse"])
    low, high, better = bootstrap_interval(run_errors, reference_errors, seed)
    return {
        "run_step": steps[0],
        "reference_step": steps[1],
        "windows": len(run_errors),
        "difference": float(np.sqrt(run_errors.mean()) - np.sqrt(reference_errors.mean())),
        "low": low,
        "high": high,
        "fraction_run_better": better,
    }


def compare_runs(runs_root, run, reference, key=HEADLINE, reference_root=None, at_final=False):
    run_records = evaluations_of(runs_root, run, key)
    reference_records = evaluations_of(reference_root or runs_root, reference, key)
    level = reference_records[-1][key]
    return {
        "run": run,
        "reference": reference,
        "metric": key,
        "target": level,
        "crossing": crossing_record(run_records, key, level),
        "fractions": {
            "run": fraction_table(run_records, key),
            "reference": fraction_table(reference_records, key),
        },
        "curve_area": {
            "run": curve_area(run_records, key),
            "reference": curve_area(reference_records, key),
        },
        "final": {"run": run_records[-1][key], "reference": reference_records[-1][key]},
        "paired": paired_bootstrap(runs_root, run, reference, at_final=at_final, reference_root=reference_root),
    }


def write_comparison(runs_root, result):
    path = Path(runs_root) / f"comparison_{result['run']}_vs_{result['reference']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def format_comparison(result):
    lines = [f"{result['run']} against {result['reference']} on {result['metric']} (lower is better)"]
    lines.append(f"  target (reference final value): {result['target']:.4f}")
    crossing = result["crossing"]
    if crossing is None:
        lines.append("  the run did not reach the target and stay there")
    else:
        hours = crossing["train_seconds"] / 3600
        lines.append(f"  reached at step {crossing['step']} ({crossing['samples_seen']} samples, {hours:.2f} h)")
    for fraction, entry in result["fractions"]["run"].items():
        other = result["fractions"]["reference"][fraction]
        lines.append(f"  at {float(fraction):4.0%} of training: {entry['value']:.4f} against {other['value']:.4f}")
    area = result["curve_area"]
    lines.append(f"  mean over the curve: {area['run']:.4f} against {area['reference']:.4f}")
    paired = result["paired"]
    if paired is not None:
        lines.append(
            f"  paired difference at step {paired['run_step']} against step {paired['reference_step']}: "
            f"{paired['difference']:+.4f} (95% interval {paired['low']:+.4f} to {paired['high']:+.4f}, "
            f"{paired['windows']} windows)"
        )
    return "\n".join(lines)
