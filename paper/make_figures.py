import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spheridiff.common.config import load_data_config
from spheridiff.data.loading import load_dataset_statistics, load_grid
from spheridiff.data.transforms import apply_precipitation_transform
from spheridiff.sphere.slepian import KernelSlepianTransform
from spheridiff.sphere.transform import SphericalTransform

FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = ROOT / "results"
GLOBAL_LIMITS = [16, 20, 24, 28, 30, 31]
REGIONAL_LIMITS = [61, 75, 90, 105, 120]
CURVES = [
    ("WIND-S", "pilot/pilot_wind_s", "black", "-"),
    ("SpherIDiff $L=16$", "pilot/pilot_spheridiff_b4", "tab:gray", ":"),
    ("SpherIDiff $L=31$", "pilot_l31/pilot_spheridiff_l31_b4", "tab:brown", "--"),
    ("Pooled", "pooled/pilot_pooled3_dense", "tab:blue", "--"),
    ("Pooled + filters + noise level", "pooled_noise_coeff/pilot_pooled3_noise_coeff", "tab:purple", "-."),
    ("Pooled + spherical convolution", "local_hybrid/pilot_pooled3_disco", "tab:red", "-"),
    ("Hybrid U-ViT", "local_hybrid/pilot_hybrid", "tab:green", "-"),
]
PATH = [
    ("$L=16$", "pilot/pilot_spheridiff_b4", "finetune/finetune_spheridiff_b4_l16"),
    ("$L=31$", "pilot_l31/pilot_spheridiff_l31_b4", "finetune/finetune_spheridiff_l31_b4"),
    ("Pooled", "pooled/pilot_pooled3_dense", "pooled/finetune_pooled3_dense"),
    ("+ filters\n+ noise level", "pooled_noise_coeff/pilot_pooled3_noise_coeff", "pooled_noise_coeff/finetune_pooled3_noise_coeff"),
    ("+ spherical\nconvolution", "local_hybrid/pilot_pooled3_disco", "local_hybrid/finetune_pooled3_disco"),
    ("Hybrid\nU-ViT", "local_hybrid/pilot_hybrid", "local_hybrid/finetune_hybrid"),
]
GLOBAL_REFERENCE = "pilot/pilot_wind_s"
REGIONAL_REFERENCE = "finetune/finetune_wind_s"
BUDGET = 25000
FINETUNE_STEPS = 5000


def evaluations(run):
    lines = (RESULTS / run / "evaluations.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    return np.array([r["step"] for r in records]), np.array([r["nrmse/mean"] for r in records])


def value_at(run, step):
    steps, values = evaluations(run)
    return float(np.interp(step, steps, values))


def normalised_frames(name, year, stride=4):
    config = load_data_config(name)
    statistics = load_dataset_statistics(config)
    raw = np.load(config.cache_root / "frames" / f"{year}.npy")[::stride].astype(np.float64)
    frames = apply_precipitation_transform(raw, statistics.names, config.precipitation_channels)
    frames = (frames - statistics.mean[None, :, None, None]) / statistics.std[None, :, None, None]
    return config, frames


def floor(transform, frames, area):
    field = torch.tensor(frames, dtype=torch.float32)
    residual = frames - transform.band_limit(field).double().numpy()
    mse = (residual**2 * area).mean(axis=(0, 2, 3)) / area.mean()
    return float(np.sqrt(mse).mean())


def global_floors():
    config, frames = normalised_frames("era5_5p6deg", 2019)
    grid = load_grid(config)
    area = grid.area_weights
    floors = []
    for limit in tqdm(GLOBAL_LIMITS, desc="global floor"):
        analysis = "quadrature" if limit <= grid.bandlimit else "least_squares"
        floors.append(floor(SphericalTransform(grid, limit, analysis), frames, area))
    return floors


def regional_floors():
    config, frames = normalised_frames("era5_1p5deg_south_asia", 2020)
    grid = load_grid(config)
    area = grid.area_weights
    return [floor(KernelSlepianTransform(grid, limit), frames, area) for limit in tqdm(REGIONAL_LIMITS, desc="regional floor")]


def floor_figure():
    globals_, regionals = global_floors(), regional_floors()
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.0, 2.8))
    left.plot(GLOBAL_LIMITS, globals_, "o-", color="tab:blue")
    left.axvline(16, color="gray", linestyle=":", label="quadrature limit")
    left.axvline(31, color="tab:red", linestyle="--", label="least-squares limit")
    left.set(xlabel="band limit $L$", ylabel="floor (nRMSE)", title="Global grid, 5.6$^\\circ$")
    right.plot(REGIONAL_LIMITS, regionals, "o-", color="tab:blue")
    right.axvline(61, color="gray", linestyle=":", label="quadrature limit")
    right.axvline(120, color="tab:red", linestyle="--", label="least-squares limit")
    right.set(xlabel="regional band limit $L'$", title="Regional patch, 1.5$^\\circ$")
    for axis in (left, right):
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_floor.pdf")
    plt.close(figure)
    return globals_, regionals


def curves_figure():
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    for label, run, colour, style in CURVES:
        steps, values = evaluations(run)
        keep = (steps > 0) & (steps <= BUDGET)
        axis.plot(steps[keep] / 1000, values[keep], style, color=colour, label=label, linewidth=1.3)
    axis.set(xlabel="training step (thousands)", ylabel="validation nRMSE", ylim=(0.22, 0.6))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_curves.pdf")
    plt.close(figure)


def gap_figure():
    global_reference = value_at(GLOBAL_REFERENCE, BUDGET)
    regional_reference = value_at(REGIONAL_REFERENCE, FINETUNE_STEPS)
    labels = [label for label, _, _ in PATH]
    global_gaps = [value_at(run, BUDGET) - global_reference for _, run, _ in PATH]
    regional_gaps = [value_at(run, FINETUNE_STEPS) - regional_reference for _, _, run in PATH]
    positions = np.arange(len(PATH))
    figure, axis = plt.subplots(figsize=(6.0, 2.8))
    axis.bar(positions - 0.2, global_gaps, 0.4, color="tab:blue", label="global, after 25,000 steps")
    axis.bar(positions + 0.2, regional_gaps, 0.4, color="tab:orange", label="regional, after fine-tuning")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, fontsize=7)
    axis.set(ylabel="nRMSE minus WIND-S")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_gap.pdf")
    plt.close(figure)


def box(axis, x, y, width, height, text, colour, size=7):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", facecolor=colour, edgecolor="black", linewidth=0.7)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size)


def arrow(axis, start, end, style="-|>", colour="black", linestyle="-"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=8, color=colour, linestyle=linestyle, linewidth=0.8))


def block_text(level, band_limit, width, local):
    lines = [f"block, $L={band_limit}$, $d={width}$", "spectral stage $T\\,W_c\\,S$", "channel MLP", "temporal attention"]
    if local:
        lines.append("spherical convolution")
    return "\n".join(lines)


def architecture_figure():
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6.0)
    axis.axis("off")
    width, height = 2.2, 1.15
    rows = [(3.6, 31, 64, "32 x 64", 1.3, 6.5, "#f7d9d4"), (1.95, 15, 128, "16 x 32", 2.2, 5.6, "#d6e6f5")]
    for index, (bottom, band_limit, channels, grid, left, right, colour) in enumerate(rows):
        text = block_text(index, band_limit, channels, index == 0)
        box(axis, left, bottom, width, height, text, colour, 6)
        box(axis, right, bottom, width, height, text, colour, 6)
        axis.text(0.05, bottom + height / 2, f"level {index}\n{grid}", fontsize=7, va="center")
        arrow(axis, (left + width, bottom + height / 2), (right, bottom + height / 2), colour="gray", linestyle="--")
        axis.text((left + width + right) / 2, bottom + height / 2 + 0.12, "skip (add)", fontsize=6, ha="center", color="gray")
    box(axis, 3.9, 0.3, width, height, block_text(2, 7, 256, False), "#dcefd8", 6)
    axis.text(0.05, 0.3 + height / 2, "level 2\n8 x 16", fontsize=7, va="center")
    box(axis, 1.3, 5.2, width, 0.55, "uplift: 35 channels to 64\n+ noise-level embedding", "#eeeeee", 6)
    box(axis, 6.5, 5.2, width, 0.55, "downlift: 64 channels to 25", "#eeeeee", 6)
    arrow(axis, (2.4, 5.2), (2.4, 4.75))
    arrow(axis, (7.6, 4.75), (7.6, 5.2))
    arrow(axis, (2.6, 3.6), (3.1, 3.1))
    axis.text(1.55, 3.25, "pool: keep $\\ell < 15$", fontsize=6)
    arrow(axis, (3.5, 1.95), (4.3, 1.45))
    axis.text(2.45, 1.6, "pool: keep $\\ell < 7$", fontsize=6)
    arrow(axis, (5.7, 1.45), (6.5, 1.95))
    axis.text(6.45, 1.6, "unpool", fontsize=6)
    arrow(axis, (6.9, 3.1), (7.4, 3.6))
    axis.text(7.35, 3.25, "unpool", fontsize=6)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_architecture.pdf")
    plt.close(figure)


def main():
    FIGURES.mkdir(exist_ok=True)
    architecture_figure()
    curves_figure()
    gap_figure()
    globals_, regionals = floor_figure()
    print("global floors", dict(zip(GLOBAL_LIMITS, np.round(globals_, 4))))
    print("regional floors", dict(zip(REGIONAL_LIMITS, np.round(regionals, 4))))


if __name__ == "__main__":
    main()
