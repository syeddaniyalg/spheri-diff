import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.experiments.spec import load_experiment
from spheridiff.finetune.transfer import apply_adaptation, transfer_weights
from spheridiff.grid.grid import GridSpec
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spec import load_model_spec
from spheridiff.models.spheridiff.local import BASIS_FUNCTIONS, LocalStage, kernel_matrix

PATCH = GridSpec(18.0 + 1.5 * np.arange(16), 58.5 + 1.5 * np.arange(16))
SMALL_PATCH = GridSpec(10.0 + 3.0 * np.arange(8), 40.0 + 3.0 * np.arange(16))
SMALL_GLOBAL = {"bandlimit": 7, "analysis": "least_squares"}
SMALL_REGIONAL = {"kind": "slepian", "method": "kernel", "bandlimit": 24}
SMALL_POOLED = {"base_width": 8, "levels": 3, "heads": 4, "spectral_precision": "fp16", "noise_conditioning": True, "spherical_convolution": True}
SMALL_HYBRID = {"base_width": 32, "levels": 3, "heads": 4, "embedding_dim": 16, "norm_groups": 8, "transformer_blocks": 1}


def small(name, architecture, grid=None, sphere=None):
    return build_model(name, architecture, ModelContext(6, 3, 5, grid or make_grid(8, 16), sphere or SMALL_GLOBAL))


def randomised(model, deviation=0.05):
    torch.manual_seed(0)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.normal_(0.0, deviation)
    return model


def test_the_spherical_convolution_is_the_same_operator_on_both_grids():
    global_kernel = kernel_matrix(make_grid(32, 64), regional=False)
    patch_kernel = kernel_matrix(PATCH, regional=True)
    assert global_kernel.shape == (BASIS_FUNCTIONS * 2048, 2048)
    assert abs((global_kernel[:2048] @ np.ones(2048)).mean() - 1.0) < 0.03
    centre = (np.abs(PATCH.latitudes[:, None] - 29.0) < 3.0) & (np.abs(PATCH.longitudes[None, :] - 70.0) < 3.0)
    assert abs((patch_kernel[:256] @ np.ones(256)).reshape(16, 16)[centre].mean() - 1.0) < 0.03


def test_the_direction_basis_cancels_on_a_constant_field_away_from_edges():
    kernel = kernel_matrix(make_grid(32, 64), regional=False).reshape(BASIS_FUNCTIONS, 2048, 2048)
    equator = np.arange(2048).reshape(32, 64)[14:18].ravel()
    response = kernel[2:, equator] @ np.ones(2048)
    assert np.abs(response).max() < 0.05


def test_the_local_stage_starts_at_zero_and_learns():
    stage = LocalStage(make_grid(8, 16), 4)
    values = torch.randn(2, 5, 128, 4, requires_grad=True)
    output = stage(values)
    assert output.shape == values.shape and torch.count_nonzero(output) == 0
    output.square().sum().backward()
    assert stage.convolution.weight.grad is None or torch.count_nonzero(stage.convolution.weight.grad) == 0
    stage.convolution.weight.data.normal_()
    stage(values).square().sum().backward()
    assert torch.count_nonzero(stage.convolution.weight.grad) > 0


def test_the_spherical_convolution_is_added_only_at_the_finest_level():
    model = small("spheridiff_pooled", SMALL_POOLED)
    assert isinstance(model.down[0].local, LocalStage) and isinstance(model.up[0].local, LocalStage)
    assert model.down[1].local is None and model.middle.local is None and model.up[1].local is None


@pytest.mark.parametrize("name, expected", [("spheridiff_pooled3_disco", 2958206), ("spheridiff_hybrid", 2685308)])
def test_the_two_variants_have_the_planned_sizes(name, expected):
    spec = load_model_spec(name)
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == expected
    regional = load_model_spec(name + "_regional")
    assert regional.architecture == spec.architecture and regional.sphere["bandlimit"] == 120


def test_the_hybrid_gives_most_parameters_to_the_wind_style_blocks():
    spec = load_model_spec("spheridiff_hybrid")
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    total = count_parameters(model)
    spectral = sum(p.numel() for n, p in model.named_parameters() if ".spectral." in n)
    wind = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("down_blocks", "up_blocks", "transformers")))
    assert wind / total > 0.55 and 0.3 < spectral / total < 0.45


def test_the_hybrid_starts_at_zero_and_uses_the_frame_levels():
    model = small("spheridiff_hybrid", SMALL_HYBRID)
    noisy, auxiliary = torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16)
    assert torch.count_nonzero(model(noisy, auxiliary)) == 0
    model = randomised(model)
    model.set_frame_levels(torch.zeros(1, 5))
    first = model(noisy, auxiliary)
    model.set_frame_levels(torch.ones(1, 5))
    assert not torch.allclose(first, model(noisy, auxiliary))


def test_the_hybrid_shares_one_spectral_weight_per_level():
    model = small("spheridiff_hybrid", SMALL_HYBRID)
    weights = [name for name, _ in model.named_parameters() if name.endswith("spectral.weight")]
    assert weights == ["mixers.0.spectral.weight", "mixers.1.spectral.weight", "mixers.2.spectral.weight"]


@pytest.mark.parametrize("name, architecture", [("spheridiff_pooled", SMALL_POOLED), ("spheridiff_hybrid", SMALL_HYBRID)])
def test_both_variants_transfer_to_the_patch(name, architecture):
    source = randomised(small(name, architecture))
    weights = {key: parameter.detach().clone() for key, parameter in source.named_parameters()}
    regional = apply_adaptation(small(name, architecture, SMALL_PATCH, SMALL_REGIONAL), "delta_and_pointwise")
    transfer_weights(regional, weights)
    regional.set_frame_levels(torch.rand(1, 5))
    output = regional(torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16))
    assert torch.isfinite(output).all()
    assert any(key.endswith("spectral.delta") for key, _ in regional.named_parameters())


@pytest.mark.parametrize("variant", ["pooled3_disco", "hybrid"])
def test_the_experiments_pair_pretraining_and_fine_tuning(variant):
    pretraining, finetuning = load_experiment(f"pilot_{variant}"), load_experiment(f"finetune_{variant}")
    assert (pretraining.train, finetuning.train) == ("pilot_25k", "finetune")
    assert finetuning.pretrained == f"pilot_{variant}/ema_weights.pt"
    assert finetuning.model == pretraining.model + "_regional"
