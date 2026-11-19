import json
import os
from pathlib import Path

import torch


class RunPaths:
    def __init__(self, root, name):
        self.root = Path(root) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.steps = self.root / "steps.jsonl"
        self.evaluations = self.root / "evaluations.jsonl"
        self.checkpoint = self.root / "checkpoint.pt"
        self.final = self.root / "final.pt"
        self.weights = self.root / "ema_weights.pt"
        self.summary = self.root / "summary.json"
        self.snapshot = self.root / "config_snapshot.json"


def append_record(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def read_records(path):
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate_records(path, step):
    records = [record for record in read_records(path) if record["step"] <= step]
    Path(path).unlink(missing_ok=True)
    for record in records:
        append_record(path, record)


def write_summary(path, summary):
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)


def save_atomic(path, state):
    temporary = Path(str(path) + ".part")
    torch.save(state, temporary)
    os.replace(temporary, path)
