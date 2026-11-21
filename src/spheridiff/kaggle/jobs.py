from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    script: str
    arguments: tuple
    budgeted: bool = False


def assign_queues(groups, gpu_count):
    if gpu_count >= len(groups):
        return [list(group) for group in groups]
    return [[job for group in groups for job in group]]


def place_queues(placements, gpu_count):
    return [(gpu % gpu_count, list(jobs)) for gpu, jobs in placements]


def train_job(experiment):
    return Job(experiment, "train.py", ("--experiment", experiment), budgeted=True)


def final_job(experiment, protocol="final_v1", run_name=None):
    arguments = ("--experiments", experiment, "--protocol", protocol)
    if run_name is not None:
        arguments += ("--run-names", run_name)
    return Job(f"final_{run_name or experiment}", "final_evaluation.py", arguments)


def finetune_job(experiment, pretrained_root):
    arguments = ("--experiment", experiment, "--pretrained-root", str(pretrained_root))
    return Job(experiment, "finetune.py", arguments, budgeted=True)


def cooldown_job(experiment, milestones_root, starts, fraction):
    arguments = (
        "--experiment",
        experiment,
        "--milestones-root",
        str(milestones_root),
        "--starts",
        *(str(step) for step in starts),
        "--fraction",
        str(fraction),
        "--keep-only-weights",
    )
    return Job(f"cooldown_{experiment}", "cooldown.py", arguments)


def smoke_job(experiment):
    return Job(experiment, "smoke.py", ("--experiment", experiment))
