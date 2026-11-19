import gc
import json

import torch
from tqdm import tqdm

from spheridiff.experiments.spec import load_experiment
from spheridiff.training.runlog import RunPaths, write_summary

from .stages import STAGES


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_stage(name, spec, device, runs_root, cache_root):
    tqdm.write(f"stage {name}")
    result = STAGES[name](spec, device, runs_root, cache_root)
    release_memory()
    tqdm.write(f"stage {name}: {json.dumps(result, default=float)}")
    return result


def run_smoke(experiment, device, runs_root, cache_root=None, stages=tuple(STAGES)):
    spec = load_experiment(experiment)
    report = {"experiment": spec.name, "device": str(device)}
    for name in tqdm(stages, desc=f"smoke {spec.name}"):
        report[name] = run_stage(name, spec, device, runs_root, cache_root)
    report["passed"] = all(report[name]["passed"] for name in stages)
    write_summary(RunPaths(runs_root, spec.name).root / "smoke_report.json", report)
    return report
