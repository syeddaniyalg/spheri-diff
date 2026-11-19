import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .monitor import LogTail, format_report

GRACE_SECONDS = 600
MINIMUM_SECONDS = 300
KILL_WAIT_SECONDS = 30


@dataclass(frozen=True)
class RunContext:
    code_root: Path
    runs_root: Path
    log_dir: Path
    cache_root: Path
    deadline: float
    grace_seconds: float = GRACE_SECONDS
    minimum_seconds: float = MINIMUM_SECONDS


def job_command(job, context, now):
    command = [sys.executable, str(context.code_root / "scripts" / job.script), *job.arguments]
    command += ["--runs-root", str(context.runs_root), "--cache-root", str(context.cache_root)]
    if job.budgeted:
        command += ["--max-seconds", str(int(context.deadline - now - context.grace_seconds))]
    return command


def job_environment(context, gpu):
    return {
        **os.environ,
        "PYTHONPATH": str(context.code_root / "src"),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TQDM_MININTERVAL": "20",
        "PYTHONUNBUFFERED": "1",
    }


class QueueWorker:
    def __init__(self, gpu, jobs, context):
        self.gpu, self.jobs, self.context = gpu, list(jobs), context
        self.process, self.name, self.log, self.started = None, None, None, 0.0
        self.outcomes = []

    @property
    def done(self):
        return self.process is None and not self.jobs

    def advance(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.finish_current()
        while self.process is None and self.jobs:
            self.start_next()

    def finish_current(self):
        if self.process is None:
            return
        self.log.close()
        seconds = time.time() - self.started
        self.outcomes.append(
            {"job": self.name, "gpu": self.gpu, "returncode": self.process.returncode, "seconds": seconds}
        )
        self.process = None

    def start_next(self):
        job = self.jobs.pop(0)
        now = time.time()
        if self.context.deadline - now < self.context.minimum_seconds + self.context.grace_seconds:
            self.outcomes.append({"job": job.name, "gpu": self.gpu, "returncode": None, "seconds": 0.0})
            return
        self.log = open(self.context.log_dir / f"{job.name}.log", "w")
        self.process = subprocess.Popen(
            job_command(job, self.context, now),
            env=job_environment(self.context, self.gpu),
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.name, self.started = job.name, now

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.finish_current()
        for job in self.jobs:
            self.outcomes.append({"job": job.name, "gpu": self.gpu, "returncode": None, "seconds": 0.0})
        self.jobs = []


class ParallelRunner:
    def __init__(self, queues, context, report_every=60, poll_every=5, refresh=None, refresh_every=1800):
        context.log_dir.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.workers = [QueueWorker(gpu, jobs, context) for gpu, jobs in enumerate(queues)]
        self.report_every, self.poll_every = report_every, poll_every
        self.refresh, self.refresh_every = refresh, refresh_every
        self.started = time.time()
        self.combined = context.log_dir / "parallel.log"

    def report(self):
        lines = []
        for worker in self.workers:
            if worker.name is not None:
                line = LogTail(self.context.log_dir / f"{worker.name}.log").last_line()
                lines.append(format_report(worker.name, worker.gpu, time.time() - self.started, line))
        if not lines:
            return
        tqdm.write("\n".join(lines))
        with open(self.combined, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def outcomes(self):
        return [outcome for worker in self.workers for outcome in worker.outcomes]

    def stop_all(self):
        tqdm.write("the time limit was reached, stopping the running jobs")
        for worker in self.workers:
            worker.stop()

    def run(self):
        next_report, next_refresh = 0.0, time.time() + self.refresh_every
        try:
            while not all(worker.done for worker in self.workers):
                for worker in self.workers:
                    worker.advance()
                now = time.time()
                if now >= next_report:
                    self.report()
                    next_report = now + self.report_every
                if self.refresh is not None and now >= next_refresh:
                    self.refresh()
                    next_refresh = now + self.refresh_every
                if now >= self.context.deadline:
                    self.stop_all()
                    break
                time.sleep(self.poll_every)
        finally:
            for worker in self.workers:
                worker.stop()
        return self.outcomes()
