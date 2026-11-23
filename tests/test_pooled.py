import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.common.config import CONFIG_ROOT
from spheridiff.experiments.spec import load_experiment
from spheridiff.finetune.transfer import apply_adaptation, transfer_weights
from spheridiff.grid.grid import GridSpec
from spheridiff.kaggle.jobs import final_job, finetune_job, train_job
from spheridiff.kaggle.runner import RunContext, job_command
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spec import load_model_spec
from spheridiff.models.spheridiff.pooled import PooledConfig
from spheridiff.sphere.slepian import KernelSlepianTransform, SlepianTransform
from spheridiff.sphere.transform import build_transform

GLOBAL = {"bandlimit": 31, "analysis": "least_squares"}
SIZES = {"pooled3_dense": 2658238, "pooled3_rank32": 864910, "pooled4_dense": 2674448, "pooled4_rank32": 882408}
SMALL_PATCH = GridSpec(10.0 + 3.0 * np.arange(8), 40.0 + 3.0 * np.arange(16))
SMALL_REGIONAL = {"kind": "slepian", "method": "kernel", "bandlimit": 24}
SMALL_GLOBAL = {"bandlimit": 7, "analysis": "least_squares"}
SMALL_ARCHITECTURE = {"base_width": 8, "levels": 3, "heads": 4, "spectral_precision": "fp16"}


def small_global(rank=None):
    architecture = {**SMALL_ARCHITECTURE, "rank": rank}
    return build_model("spheridiff_pooled", architecture, ModelContext(6, 3, 5, make_grid(8, 16), SMALL_GLOBAL))


def small_regional(adaptation="delta_and_pointwise"):
    model = build_model("spheridiff_pooled", SMALL_ARCHITECTURE, ModelContext(6, 3, 5, SMALL_PATCH, SMALL_REGIONAL))
    return apply_adaptation(model, adaptation)


@pytest.mark.parametrize("name", sorted(SIZES))
def test_pooled_configs_have_the_planned_sizes_and_band_limits(name):
    spec = load_model_spec(f"spheridiff_{name}")
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == SIZES[name]
    expected = [31, 15, 7, 3][: spec.architecture["levels"]]
    assert [transform.degrees for transform in model.plan.transforms] == expected
    assert spec.architecture["spectral_precision"] == "fp16"


def test_the_global_pooling_keeps_band_limited_fields():
    model = small_global()
    coarse = model.plan.transforms[1]
    field = coarse.synthesise_points(torch.randn(coarse.coefficients, 2))
    assert torch.allclose(model.plan.pools[0](model.plan.unpools[0](field)), field, atol=1e-4)


def test_the_regional_pooling_keeps_fields_of_the_coarse_basis():
    model = small_regional()
    coarse = model.plan.transforms[1]
    assert [t.bandlimit for t in model.plan.transforms] == [24, 12, 6]
    field = coarse.synthesise_points(torch.randn(coarse.coefficients, 2))
    assert torch.allclose(model.plan.pools[0](model.plan.unpools[0](field)), field, atol=1e-4)


def test_the_kernel_slepian_transform_spans_the_same_modes_as_the_matrix_one():
    matrix, kernel = SlepianTransform(SMALL_PATCH, 11), KernelSlepianTransform(SMALL_PATCH, 11)
    first = matrix.synthesis.T @ matrix.analysis.T
    second = kernel.synthesis.T @ kernel.analysis.T
    assert matrix.coefficients == kernel.coefficients
    assert torch.allclose(first, second, atol=1e-4)
    assert build_transform(SMALL_PATCH, {"kind": "slepian", "method": "kernel", "bandlimit": 11}).coefficients == kernel.coefficients


def test_half_precision_reaches_every_spectral_stage_and_resample():
    model = small_global()
    stages = [module for module in model.modules() if hasattr(module, "mix")]
    assert stages and all(stage.half_precision for stage in stages)
    assert all(resample.half_precision for resample in [*model.plan.pools, *model.plan.unpools])
    with pytest.raises(ValueError):
        PooledConfig(8, 3, spectral_precision="fp8")
    with pytest.raises(ValueError):
        PooledConfig(8, 1)


@pytest.mark.parametrize("rank", [None, 4])
def test_global_pooled_weights_transfer_to_the_regional_model(rank):
    torch.manual_seed(0)
    source = small_global(rank)
    for parameter in source.parameters():
        parameter.data.normal_()
    weights = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    model = transfer_weights(small_regional(), weights)
    assert torch.equal(model.uplift.weight, weights["uplift.weight"])
    assert all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if n.endswith("spectral.delta"))
    assert all(torch.count_nonzero(p) > 0 for n, p in model.named_parameters() if n.endswith("spectral.restricted"))
    output = model(torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16))
    assert output.shape == (1, 5, 6, 8, 16) and torch.isfinite(output).all()


def test_a_job_can_name_its_own_data_cache(tmp_path):
    context = RunContext(tmp_path, tmp_path / "runs", tmp_path / "logs", tmp_path / "global")
    default = job_command(train_job("a"), context, 0.0)
    own = job_command(finetune_job("b", tmp_path, tmp_path / "regional"), context, 0.0)
    final = job_command(final_job("c", "final_v1", "c_zeroshot", tmp_path / "regional"), context, 0.0)
    assert default[default.index("--cache-root") + 1] == str(tmp_path / "global")
    assert own[own.index("--cache-root") + 1] == str(tmp_path / "regional")
    assert final[final.index("--cache-root") + 1] == str(tmp_path / "regional")


@pytest.mark.parametrize("name", sorted(SIZES))
def test_pooled_experiments_pair_pretraining_and_fine_tuning(name):
    pretraining, finetuning = load_experiment(f"pilot_{name}"), load_experiment(f"finetune_{name}")
    assert (pretraining.data, pretraining.train, pretraining.seed) == ("era5_5p6deg", "pilot_25k", 42)
    assert (finetuning.data, finetuning.train, finetuning.seed) == ("era5_1p5deg_south_asia", "finetune", 42)
    assert finetuning.pretrained == f"pilot_{name}/ema_weights.pt"
    assert finetuning.model == f"spheridiff_{name}_regional"
    regional = load_model_spec(finetuning.model)
    assert regional.sphere == {"kind": "slepian", "method": "kernel", "bandlimit": 120}
    assert regional.architecture == load_model_spec(pretraining.model).architecture
    assert (CONFIG_ROOT / "model" / f"{finetuning.model}.yaml").exists()
