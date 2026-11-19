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


def train_job(experiment):
    return Job(experiment, "train.py", ("--experiment", experiment), budgeted=True)


def smoke_job(experiment):
    return Job(experiment, "smoke.py", ("--experiment", experiment))
