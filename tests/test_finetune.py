import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import xarray
import yaml

from helpers import make_grid
from pipeline import TINY_SPHERIDIFF, TINY_WIND, dump, write_experiment
from spheridiff.common.config import CONFIG_ROOT, PROJECT_ROOT, GridConfig, load_data_config
from spheridiff.data.layout import CacheLayout
from spheridiff.data.source import crop_region
from spheridiff.experiments.build import build_trainer
from spheridiff.experiments.spec import load_experiment
from spheridiff.finetune.transfer import (
    apply_adaptation,
    dense_spectral_weight,
    pretrained_weights,
    restrict,
    transfer_weights,
)
from spheridiff.grid.grid import GridSpec
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.sphere.slepian import SlepianTransform, region_area, shannon_number
from spheridiff.training.runlog import RunPaths, read_records

CPU = torch.device("cpu")
PATCH = GridSpec(18.0 + 1.5 * np.arange(16), 58.5 + 1.5 * np.arange(16))
SMALL_PATCH = GridSpec(10.0 + 3.0 * np.arange(8), 40.0 + 3.0 * np.arange(16))
SMALL_SPHERE = {"kind": "slepian", "bandlimit": 11}
REGIONAL_SPHERIDIFF = {**TINY_SPHERIDIFF, "sphere": SMALL_SPHERE}
EXPERIMENTS = [path.stem for path in (CONFIG_ROOT / "experiments").glob("finetune_*.yaml")]


def small_model(architecture=None, adaptation="delta_and_pointwise"):
    context = ModelContext(6, 3, 5, SMALL_PATCH, SMALL_SPHERE)
    model = build_model("spheridiff", architecture or {"widths": [16, 32, 16], "heads": 4}, context)
    return apply_adaptation(model, adaptation)


def test_the_patch_needs_45_slepian_modes():
    area = region_area((17.25, 41.25), (57.75, 81.75))
    assert shannon_number(area, 61) == 45
    assert abs(area / (4.0 * np.pi) * 61**2 - 45.0) < 0.01


def test_slepian_transform_inverts_and_keeps_unit_noise():
    transform = SlepianTransform(SMALL_PATCH, 11)
    eigenvalues = transform.eigenvalues
    assert transform.coefficients == shannon_number(transform.area, 11)
    assert np.all(np.diff(eigenvalues) <= 1e-12) and 0.0 <= eigenvalues.min() and eigenvalues.max() <= 1.0 + 1e-9
    coefficients = torch.randn(4, transform.coefficients)
    assert torch.allclose(transform.analyse(transform.synthesise(coefficients)), coefficients, atol=1e-4)
    noise = transform.synthesise(torch.randn(4000, transform.coefficients) * transform.noise_variance**0.5)
    assert 0.6 < noise.var().item() < 1.2


def test_restriction_of_the_identity_is_the_identity():
    transform = SlepianTransform(SMALL_PATCH, 11)
    restricted = restrict(torch.eye(121, dtype=torch.float64), transform.basis)
    assert torch.allclose(restricted, torch.eye(transform.coefficients, dtype=torch.float64), atol=1e-5)
    with pytest.raises(ValueError):
        restrict(torch.eye(144, dtype=torch.float64), transform.basis)


def test_transfer_fills_the_restricted_weight_and_copies_the_rest():
    torch.manual_seed(0)
    source = build_model("spheridiff", {"widths": [16, 32, 16], "heads": 4}, ModelContext(6, 3, 5, make_grid(8, 16)))
    for parameter in source.parameters():
        parameter.data.normal_()
    weights = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    model = transfer_weights(small_model(), weights)
    stage = model.blocks[0].spectral
    expected = restrict(dense_spectral_weight(weights, "blocks.0.spectral"), stage.transform.basis)
    assert torch.allclose(stage.restricted.double(), expected, atol=1e-5)
    assert torch.count_nonzero(stage.delta) == 0
    assert torch.equal(model.uplift.weight, weights["uplift.weight"])


def test_structured_pretrained_weights_are_read_as_their_dense_matrix():
    weights = {"s.diagonal": torch.ones(4), "s.left": torch.ones(4, 2), "s.right": torch.ones(4, 2)}
    assert torch.allclose(dense_spectral_weight(weights, "s"), torch.eye(4, dtype=torch.float64) + 2.0)


def test_adaptations_choose_the_trainable_weights():
    full = small_model()
    delta_only = small_model(adaptation="delta_only")
    assert all(name.endswith("spectral.delta") for name, p in delta_only.named_parameters() if p.requires_grad)
    assert count_parameters(delta_only) == 2 * delta_only.blocks[0].spectral.transform.coefficients**2
    assert count_parameters(full) > count_parameters(delta_only)
    assert not any(p.requires_grad for name, p in full.named_parameters() if name.endswith("restricted"))
    wind = build_model("wind", TINY_WIND["architecture"], ModelContext(6, 3, 5, make_grid(8, 16)))
    with pytest.raises(ValueError):
        apply_adaptation(wind, "delta_only")
    with pytest.raises(ValueError):
        apply_adaptation(wind, "partial")


def test_wind_weights_fit_a_patch_of_another_size():
    global_model = build_model("wind", TINY_WIND["architecture"], ModelContext(6, 3, 5, make_grid(8, 16)))
    weights = {name: parameter.detach().clone() for name, parameter in global_model.named_parameters()}
    patch_model = build_model("wind", TINY_WIND["architecture"], ModelContext(6, 3, 5, SMALL_PATCH))
    transfer_weights(patch_model, weights)
    output = patch_model(torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16))
    assert output.shape == (1, 5, 6, 8, 16)


def test_the_region_is_cropped_from_the_global_store():
    latitudes, longitudes = np.linspace(-90.0, 90.0, 121), np.arange(240) * 1.5
    dataset = xarray.Dataset(coords={"latitude": latitudes, "longitude": longitudes})
    grid = GridConfig(16, 16, (18.0, 40.5), (58.5, 81.0))
    cropped = crop_region(dataset, grid)
    assert np.allclose(cropped["latitude"].values, PATCH.latitudes)
    assert np.allclose(cropped["longitude"].values, PATCH.longitudes)
    with pytest.raises(ValueError):
        crop_region(dataset, GridConfig(15, 16, (18.0, 40.5), (58.5, 81.0)))
    assert crop_region(dataset, GridConfig(121, 240)) is dataset


def test_the_regional_data_config_describes_the_patch():
    config = load_data_config("era5_1p5deg_south_asia")
    assert config.grid.regional and (config.grid.height, config.grid.width) == (16, 16)
    assert config.grid.latitude_range == (18.0, 40.5) and config.grid.longitude_range == (58.5, 81.0)
    assert config.channels == load_data_config("era5_5p6deg").channels


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_finetune_experiments_name_their_start_and_adaptation(name):
    spec = load_experiment(name)
    assert spec.data == "era5_1p5deg_south_asia" and spec.train == "finetune" and spec.seed == 42
    assert spec.pretrained.endswith(".pt")
    expected = "full" if "wind" in name else ("delta_only" if name.endswith("delta_only") else "delta_and_pointwise")
    assert spec.adaptation == expected


def regional_cache(directory):
    experiment, cache_root = write_experiment(directory, TINY_WIND, steps=2)
    layout = CacheLayout(cache_root)
    static = dict(np.load(layout.static_path))
    static["latitudes"], static["longitudes"] = SMALL_PATCH.latitudes, SMALL_PATCH.longitudes
    np.savez(layout.static_path, **static)
    return cache_root


def test_the_finetune_script_saves_the_zero_shot_and_the_fine_tuned_run(tmp_path):
    experiment, cache_root = write_experiment(tmp_path / "global", TINY_SPHERIDIFF, steps=4)
    pretrained, _ = build_trainer(load_experiment(experiment), CPU, tmp_path / "pretrained", cache_root)
    pretrained.run(resume=False)
    regional_root = regional_cache(tmp_path / "regional")
    spec = yaml.safe_load(open(experiment))
    spec.update({
        "name": "tiny_finetune",
        "data": str(tmp_path / "regional" / "data.yaml"),
        "model": dump(tmp_path / "regional_model.yaml", REGIONAL_SPHERIDIFF),
        "pretrained": "tiny/ema_weights.pt",
        "adaptation": "delta_and_pointwise",
    })
    path = dump(tmp_path / "finetune.yaml", spec)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "finetune.py"),
        "--experiment", path, "--pretrained-root", str(tmp_path / "pretrained"),
        "--runs-root", str(tmp_path / "runs"), "--cache-root", str(regional_root),
    ]
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    zero_shot = torch.load(RunPaths(tmp_path / "runs", "tiny_finetune_zeroshot").weights, weights_only=False)
    assert zero_shot["step"] == 0
    fine_tuned = RunPaths(tmp_path / "runs", "tiny_finetune")
    assert [record["step"] for record in read_records(fine_tuned.steps)][-1] == 4
    assert json.loads(fine_tuned.summary.read_text())["steps"] == 4
