import json
import time
import zipfile

import pytest

from spheridiff.kaggle.finalise import finalise
from spheridiff.kaggle.jobs import Job, assign_queues, smoke_job, train_job
from spheridiff.kaggle.monitor import LogStream, format_report
from spheridiff.kaggle.runner import ParallelRunner, RunContext, job_command

DUMMY = """import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
print("gpu", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
print("args", sys.argv[1:], flush=True)
print("bar \\u258d \\u2588 0%", flush=True)
time.sleep(float(sys.argv[1]))
sys.exit(int(sys.argv[2]))
"""


@pytest.fixture
def context(tmp_path):
    scripts = tmp_path / "code" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "dummy.py").write_text(DUMMY)
    (tmp_path / "code" / "src").mkdir()
    return RunContext(
        tmp_path / "code",
        tmp_path / "runs",
        tmp_path / "runs" / "_logs",
        tmp_path / "cache",
        time.time() + 600,
        grace_seconds=0.0,
        minimum_seconds=0.0,
    )


def dummy_job(name, seconds, code=0):
    return Job(name, "dummy.py", (str(seconds), str(code)))


def test_one_group_per_gpu_when_enough_gpus():
    groups = [[Job("a", "x", ())], [Job("b", "x", ())]]
    assert [[job.name for job in queue] for queue in assign_queues(groups, 2)] == [["a"], ["b"]]


def test_all_jobs_share_one_queue_with_a_single_gpu():
    groups = [[Job("a", "x", ()), Job("c", "x", ())], [Job("b", "x", ())]]
    queues = assign_queues(groups, 1)
    assert len(queues) == 1
    assert [job.name for job in queues[0]] == ["a", "c", "b"]


def test_job_factories_use_the_right_scripts():
    assert train_job("pilot_wind_s").script == "train.py" and train_job("pilot_wind_s").budgeted
    assert smoke_job("smoke_wind_s").script == "smoke.py" and not smoke_job("smoke_wind_s").budgeted


def test_budgeted_jobs_receive_the_remaining_time(context):
    budgeted = Job("t", "train.py", ("--experiment", "e"), budgeted=True)
    command = job_command(budgeted, context, time.time())
    assert command[command.index("--max-seconds") + 1].isdigit()
    assert "--max-seconds" not in job_command(Job("s", "smoke.py", ()), context, time.time())


def test_queues_run_in_parallel_with_their_own_gpu(context):
    queues = [[dummy_job("first", 2.0), dummy_job("third", 0.2)], [dummy_job("second", 2.0)]]
    started = time.time()
    runner = ParallelRunner(queues, context, poll_every=0.1)
    outcomes = runner.run()
    elapsed = time.time() - started
    assert elapsed < 4.0
    assert {outcome["job"]: outcome["returncode"] for outcome in outcomes} == {
        "first": 0, "second": 0, "third": 0,
    }
    assert "gpu 0" in (context.log_dir / "first.log").read_text(encoding="utf-8")
    assert "gpu 0" in (context.log_dir / "third.log").read_text(encoding="utf-8")
    assert "gpu 1" in (context.log_dir / "second.log").read_text(encoding="utf-8")
    combined = (context.log_dir / "parallel.log").read_text(encoding="utf-8")
    assert "[gpu 0] [first]" in combined and "[gpu 1] [second]" in combined
    for name, gpu in (("first", 0), ("second", 1), ("third", 0)):
        for fragment in (f"gpu {gpu}", "args", "bar ▍ █ 0%"):
            assert f"[{name}] {fragment}" in combined


def test_a_failing_job_does_not_stop_the_others(context):
    runner = ParallelRunner([[dummy_job("bad", 0.2, code=3)], [dummy_job("good", 0.5)]], context, poll_every=0.1)
    codes = {outcome["job"]: outcome["returncode"] for outcome in runner.run()}
    assert codes == {"bad": 3, "good": 0}


def test_the_deadline_stops_running_jobs(context):
    short = RunContext(
        context.code_root, context.runs_root, context.log_dir, context.cache_root,
        time.time() + 1.5, grace_seconds=0.0, minimum_seconds=0.0,
    )
    runner = ParallelRunner([[dummy_job("slow", 60.0), dummy_job("never", 1.0)]], short, poll_every=0.1)
    started = time.time()
    outcomes = {outcome["job"]: outcome["returncode"] for outcome in runner.run()}
    assert time.time() - started < 15
    assert outcomes["slow"] != 0 and outcomes["never"] is None


def test_the_archive_is_written_after_a_failure(context, tmp_path):
    runner = ParallelRunner([[dummy_job("bad", 0.1, code=1)]], context, poll_every=0.1)
    outcomes = []
    archive = tmp_path / "results.zip"
    try:
        outcomes = runner.run()
        raise RuntimeError("the notebook cell failed")
    except RuntimeError:
        pass
    finally:
        finalise(context.runs_root, archive, outcomes, ["bad"])
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    assert "status.json" in names and "_logs/bad.log" in names
    assert json.loads((context.runs_root / "status.json").read_text())[0]["returncode"] == 1


def test_log_stream_returns_every_new_line_once(tmp_path):
    path = tmp_path / "job.log"
    stream = LogStream(path)
    assert stream.read() == []
    path.write_bytes(b"start\nstep 1\rstep 2\rste")
    assert stream.read() == ["start", "step 1", "step 2"]
    with open(path, "ab") as handle:
        handle.write(b"p 3\rstep 4\n")
    assert stream.read() == ["step 3", "step 4"]
    with open(path, "ab") as handle:
        handle.write(b"last without ending")
    assert stream.read() == []
    assert stream.flush() == ["last without ending"]
    assert stream.read() == []
    assert "[gpu 1]" in format_report("job", 1, 120.0, "text")
