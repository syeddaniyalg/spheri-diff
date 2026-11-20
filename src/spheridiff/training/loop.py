import statistics
import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from spheridiff.common.seed import DEFAULT_SEED
from spheridiff.data.batches import random_batches
from spheridiff.diffusion.objective import denoising_loss

from .checkpoint import load_state, restore_state, trainer_state
from .ema import ExponentialMovingAverage
from .precision import autocast_context, build_scaler, resolve_precision
from .runlog import (
    append_record,
    milestone_name,
    read_records,
    save_atomic,
    truncate_records,
    write_summary,
)
from .schedule import scheduled_rate

SEED_STRIDE = 1_000_003


@dataclass
class TrainingParts:
    model: object
    process: object
    weights: object
    dataset: object
    evaluator: object = None
    metadata: dict = None


class Trainer:
    def __init__(self, parts, settings, paths, device, seed=DEFAULT_SEED, name="train"):
        self.parts, self.settings, self.paths = parts, settings, paths
        self.device, self.seed, self.name = device, seed, name
        self.model = parts.model.to(device)
        self.weights = parts.weights.to(device)
        self.precision = resolve_precision(settings.precision, device)
        self.optimizer = self.build_optimizer()
        self.scaler = build_scaler(self.precision, device)
        self.ema = ExponentialMovingAverage(self.model, settings.ema_decay)
        self.step = 0
        self.elapsed = 0.0
        self.stop_at = None
        self.milestone_steps = settings.milestone_steps

    def build_optimizer(self):
        settings = self.settings
        fused = self.device.type == "cuda"
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            fused=fused,
        )

    def run(self, resume=True, start_from=None):
        self.prepare(resume, start_from)
        self.write_snapshot()
        self.evaluate_at_start()
        self.loop()
        self.finish()

    def prepare(self, resume, start_from=None):
        state = load_state(self.paths.checkpoint, self.device) if resume else None
        if state is None and start_from is not None:
            state = load_state(start_from, self.device)
        logs = (self.paths.steps, self.paths.evaluations, self.paths.windows)
        if state is None:
            for path in logs:
                path.unlink(missing_ok=True)
            return
        restore_state(self, state)
        for path in logs:
            truncate_records(path, self.step)

    def write_snapshot(self):
        if self.parts.metadata is not None:
            write_summary(self.paths.snapshot, self.parts.metadata)

    def evaluate_at_start(self):
        if self.step == 0 and self.settings.evaluate_every > 0:
            self.evaluate(include_baselines=True)

    def loop(self):
        total = self.settings.steps
        batches = random_batches(self.parts.dataset, self.settings.batch_size, self.seed, self.step)
        progress = tqdm(range(self.step, total), initial=self.step, total=total, desc=self.name)
        started = time.perf_counter()
        for _ in progress:
            fetched = time.perf_counter()
            batch = next(batches)
            record = self.train_step(batch, time.perf_counter() - fetched)
            progress.set_postfix(self.live_metrics(record))
            self.after_step()
            if self.out_of_time(started):
                tqdm.write("time limit reached, stopping after a checkpoint")
                self.save()
                break
            if self.stop_at is not None and self.step >= self.stop_at:
                break

    def live_metrics(self, record):
        return {
            "loss": f"{record['loss']:.4f}",
            "lr": f"{record['learning_rate']:.2e}",
            "samples/s": f"{record['samples_per_second']:.1f}",
            "grad": f"{record['grad_norm']:.2f}",
        }

    def out_of_time(self, started):
        limit = self.settings.max_seconds
        return limit > 0 and time.perf_counter() - started > limit

    def step_generator(self):
        generator = torch.Generator(device=self.device)
        return generator.manual_seed(self.seed * SEED_STRIDE + self.step)

    def apply_learning_rate(self):
        value = scheduled_rate(self.settings, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def optimise(self, clean, auxiliary):
        self.optimizer.zero_grad(set_to_none=True)
        scale_before = self.scaler.get_scale()
        with autocast_context(self.precision, self.device):
            loss = denoising_loss(
                self.model, self.parts.process, self.weights, clean, auxiliary, self.step_generator()
            )
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.detach(), norm.detach(), self.scaler.get_scale() < scale_before

    def train_step(self, batch, data_seconds):
        started = time.perf_counter()
        clean = batch["frames"].to(self.device, non_blocking=True)
        auxiliary = batch["auxiliary"].to(self.device, non_blocking=True)
        learning_rate = self.apply_learning_rate()
        loss, norm, skipped = self.optimise(clean, auxiliary)
        self.ema.update(self.model)
        loss_value = self.finish_step(loss)
        seconds = time.perf_counter() - started
        return self.log_step(loss_value, norm, skipped, learning_rate, seconds, data_seconds)

    def finish_step(self, loss):
        value = loss.item()
        if not torch.isfinite(loss):
            raise RuntimeError(f"the loss became {value} at step {self.step}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return value

    def log_step(self, loss, norm, skipped, learning_rate, seconds, data_seconds):
        self.step += 1
        total = seconds + data_seconds
        self.elapsed += total
        record = {
            "step": self.step,
            "samples_seen": self.step * self.settings.batch_size,
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": float(norm),
            "skipped": bool(skipped),
            "loss_scale": self.scaler.get_scale(),
            "step_seconds": seconds,
            "data_seconds": data_seconds,
            "samples_per_second": self.settings.batch_size / total,
            "train_seconds": self.elapsed,
            "memory_gb": self.peak_memory(),
        }
        append_record(self.paths.steps, record)
        return record

    def peak_memory(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1024**3

    def after_step(self):
        settings = self.settings
        done = self.step == settings.steps
        if settings.evaluate_every > 0 and (self.step % settings.evaluate_every == 0 or done):
            self.evaluate()
        if settings.checkpoint_every > 0 and self.step % settings.checkpoint_every == 0 and not done:
            self.save()
        if self.step in self.milestone_steps:
            save_atomic(self.paths.root / milestone_name(self.step), trainer_state(self))

    def evaluate(self, include_baselines=False):
        evaluator = self.parts.evaluator
        if evaluator is None:
            return
        started = time.perf_counter()
        self.model.eval()
        with self.ema.applied(self.model), torch.no_grad():
            metrics = evaluator.evaluate(self.model)
        self.model.train()
        if include_baselines:
            metrics.update(evaluator.baselines())
        record = {
            "step": self.step,
            "samples_seen": self.step * self.settings.batch_size,
            "train_seconds": self.elapsed,
            "evaluation_seconds": time.perf_counter() - started,
            **metrics,
        }
        append_record(self.paths.evaluations, record)
        self.write_windows(evaluator)
        tqdm.write(self.describe(metrics))

    def write_windows(self, evaluator):
        record = evaluator.window_record() if hasattr(evaluator, "window_record") else None
        if record:
            append_record(self.paths.windows, {"step": self.step, **record})

    def describe(self, metrics):
        keys = [key for key in self.parts.evaluator.headline if key in metrics]
        values = ", ".join(f"{key}={metrics[key]:.4f}" for key in keys)
        return f"step {self.step}: {values}"

    def save(self):
        save_atomic(self.paths.checkpoint, trainer_state(self))
        save_atomic(self.paths.weights, {"step": self.step, "model": self.ema.shadow})

    def finish(self):
        self.save()
        if self.step >= self.settings.steps:
            save_atomic(self.paths.final, trainer_state(self))
        write_summary(self.paths.summary, self.summary())

    def summary(self):
        records = read_records(self.paths.steps)
        settled = [record["step_seconds"] + record["data_seconds"] for record in records[10:]]
        return {
            "steps": self.step,
            "train_seconds": self.elapsed,
            "median_seconds_per_step": statistics.median(settled) if settled else None,
            "final_loss": records[-1]["loss"] if records else None,
            "peak_memory_gb": self.peak_memory(),
        }
