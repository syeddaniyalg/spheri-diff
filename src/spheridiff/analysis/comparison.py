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


def common_step(first, second):
    shared = sorted(set(first) & set(second))
    return shared[-1] if shared else None


def paired_bootstrap(runs_root, run, reference, seed=DEFAULT_SEED):
    run_windows, reference_windows = window_scores(runs_root, run), window_scores(runs_root, reference)
    step = common_step(run_windows, reference_windows)
    if step is None or run_windows[step]["indices"] != reference_windows[step]["indices"]:
        return None
    errors = np.array(run_windows[step]["nmse"]), np.array(reference_windows[step]["nmse"])
    choices = np.random.default_rng(seed).integers(0, len(errors[0]), (RESAMPLES, len(errors[0])))
    resampled = np.sqrt(errors[0][choices].mean(axis=1)) - np.sqrt(errors[1][choices].mean(axis=1))
    edge = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(resampled, [edge, 1.0 - edge])
    return {
        "step": step,
        "windows": len(errors[0]),
        "difference": float(np.sqrt(errors[0].mean()) - np.sqrt(errors[1].mean())),
        "low": float(low),
        "high": float(high),
        "fraction_run_better": float((resampled < 0).mean()),
    }


def compare_runs(runs_root, run, reference, key=HEADLINE):
    run_records = evaluations_of(runs_root, run, key)
    reference_records = evaluations_of(runs_root, reference, key)
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
        "paired": paired_bootstrap(runs_root, run, reference),
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
            f"  paired difference at step {paired['step']}: {paired['difference']:+.4f} "
            f"(95% interval {paired['low']:+.4f} to {paired['high']:+.4f}, {paired['windows']} windows)"
        )
    return "\n".join(lines)
