import json
from pathlib import Path

from spheridiff.analysis.comparison import paired_bootstrap
from spheridiff.training.cooldown import branch_name
from spheridiff.training.runlog import RunPaths, read_records

KEYS = ("nrmse/mean", "crps_normalised/mean", "val_loss/mean")
HEADLINE = "nrmse/mean"


def evaluation_records(root, name):
    return [record for record in read_records(RunPaths(root, name).evaluations) if HEADLINE in record]


def values_of(record):
    return {key: record.get(key) for key in KEYS} if record else {key: None for key in KEYS}


def record_at(records, step):
    return next((record for record in records if record["step"] == step), None)


def branch_row(runs_root, reference_root, experiment, start_step, tolerance):
    name = branch_name(experiment, start_step)
    branch = evaluation_records(runs_root, name)
    reference = evaluation_records(reference_root, experiment)
    if not branch or not reference:
        return None
    paired = paired_bootstrap(runs_root, name, experiment, at_final=True, reference_root=reference_root)
    return {
        "experiment": experiment,
        "start_step": start_step,
        "end_step": branch[-1]["step"],
        "full_step": reference[-1]["step"],
        "milestone": values_of(record_at(reference, start_step)),
        "cooled": values_of(branch[-1]),
        "full": values_of(reference[-1]),
        "paired": paired,
        "matches_full": paired is not None and paired["high"] <= tolerance,
    }


def cooldown_rows(runs_root, reference_root, experiments, starts, tolerance):
    rows = []
    for experiment in experiments:
        for start_step in sorted(starts):
            row = branch_row(runs_root, reference_root, experiment, start_step, tolerance)
            if row is not None:
                rows.append(row)
    return rows


def sufficient_budget(rows, experiment):
    matching = [row["end_step"] for row in rows if row["experiment"] == experiment and row["matches_full"]]
    return min(matching) if matching else None


def number(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_row(row):
    paired = row["paired"]
    interval = "n/a"
    if paired is not None:
        interval = f"{paired['difference']:+.4f} ({paired['low']:+.4f} to {paired['high']:+.4f})"
    return (
        f"  {row['start_step']:>6} -> {row['end_step']:>6}: nRMSE {number(row['milestone'][HEADLINE])} -> "
        f"{number(row['cooled'][HEADLINE])}, CRPS {number(row['cooled']['crps_normalised/mean'])}, "
        f"val loss {number(row['cooled']['val_loss/mean'], 5)}, against step {row['full_step']}: {interval}"
        f"{', matches' if row['matches_full'] else ''}"
    )


def format_experiment(rows, experiment):
    own = [row for row in rows if row["experiment"] == experiment]
    full = own[0]["full"]
    lines = [
        f"{experiment} (full run at step {own[0]['full_step']}: nRMSE {number(full[HEADLINE])}, "
        f"CRPS {number(full['crps_normalised/mean'])}, val loss {number(full['val_loss/mean'], 5)})"
    ]
    lines += [format_row(row) for row in own]
    budget = sufficient_budget(rows, experiment)
    verdict = f"step {budget}" if budget is not None else "none of the cooled budgets"
    lines.append(f"  smallest budget that matches the full run: {verdict}")
    return "\n".join(lines)


def format_cooldown_summary(rows, tolerance):
    header = (
        "cooldown branches against the full run, paired difference of nRMSE with a 95% interval\n"
        f"a branch matches the full run when the upper end of the interval is at most {tolerance:+.4f}"
    )
    experiments = list(dict.fromkeys(row["experiment"] for row in rows))
    return "\n\n".join([header, *(format_experiment(rows, name) for name in experiments)])


def write_cooldown_summary(runs_root, reference_root, experiments, starts, tolerance):
    rows = cooldown_rows(runs_root, reference_root, experiments, starts, tolerance)
    text = format_cooldown_summary(rows, tolerance)
    Path(runs_root, "cooldown_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    Path(runs_root, "cooldown_summary.txt").write_text(text, encoding="utf-8")
    return text


def plot_cooldowns(runs_root, reference_root, experiments, starts, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(experiments), figsize=(4.5 * len(experiments), 3.6), squeeze=False)
    for axis, experiment in zip(axes[0], experiments):
        draw_experiment(axis, runs_root, reference_root, experiment, starts)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def draw_experiment(axis, runs_root, reference_root, experiment, starts):
    reference = [record for record in evaluation_records(reference_root, experiment) if record["step"] > 0]
    axis.plot([r["step"] for r in reference], [r[HEADLINE] for r in reference], color="0.5", label="cosine run")
    for start_step in sorted(starts):
        branch = evaluation_records(runs_root, branch_name(experiment, start_step))
        before = record_at(reference, start_step)
        if not branch or before is None:
            continue
        steps = [start_step, branch[-1]["step"]]
        axis.plot(steps, [before[HEADLINE], branch[-1][HEADLINE]], marker="o", label=f"cooldown from {start_step}")
    axis.set_title(experiment)
    axis.set_xlabel("training steps")
    axis.set_ylabel(HEADLINE)
    axis.legend(fontsize=7)
