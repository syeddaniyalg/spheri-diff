import numpy as np
import pytest
import torch

from helpers import make_grid
from spheridiff.diffusion.objective import denoising_loss
from spheridiff.diffusion.process import build_process
from spheridiff.diffusion.sampler import ModelDenoiser, ddim_sample, frame_levels
from spheridiff.experiments.spec import load_experiment
from spheridiff.finetune.transfer import apply_adaptation, restrict, transfer_weights
from spheridiff.grid.grid import GridSpec
from spheridiff.models.context import ModelContext
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spec import load_model_spec
from spheridiff.models.spheridiff.blocks import SpaceTimeAttention, TemporalAttention

SIZES = {"filters": 2669886, "noise": 2662526, "spacetime": 2658238}
SMALL_GLOBAL = {"bandlimit": 7, "analysis": "least_squares"}
SMALL_REGIONAL = {"kind": "slepian", "method": "kernel", "bandlimit": 24}
SMALL_PATCH = GridSpec(10.0 + 3.0 * np.arange(8), 40.0 + 3.0 * np.arange(16))
BASE = {"base_width": 8, "levels": 3, "heads": 4, "spectral_precision": "fp16"}


def small_model(option, grid=None, sphere=None):
    context = ModelContext(6, 3, 5, grid or make_grid(8, 16), sphere or SMALL_GLOBAL)
    return build_model("spheridiff_pooled", {**BASE, option: True}, context)


def randomised(model):
    torch.manual_seed(0)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.normal_(0.0, 0.1)
    return model


@pytest.mark.parametrize("name", sorted(SIZES))
def test_the_three_variants_change_only_their_own_part(name):
    spec = load_model_spec(f"spheridiff_pooled3_{name}")
    base = load_model_spec("spheridiff_pooled3_dense")
    extra = {key: value for key, value in spec.architecture.items() if key not in base.architecture}
    assert len(extra) == 1 and list(extra.values()) == [True]
    assert {k: v for k, v in spec.architecture.items() if k in base.architecture} == base.architecture
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == SIZES[name]
    pretraining, finetuning = load_experiment(f"pilot_pooled3_{name}"), load_experiment(f"finetune_pooled3_{name}")
    assert pretraining.train == "pilot_25k" and finetuning.train == "finetune"
    assert finetuning.pretrained == f"pilot_pooled3_{name}/ema_weights.pt"
    assert load_model_spec(finetuning.model).architecture == spec.architecture


def test_the_clean_context_frame_gets_level_zero():
    state = torch.zeros(2, 5, 3, 4, 8)
    mask = torch.zeros_like(state, dtype=torch.bool)
    mask[:, :1] = True
    levels = frame_levels(state, 0.7, mask)
    assert torch.allclose(levels[:, 0], torch.zeros(2)) and torch.allclose(levels[:, 1:], torch.full((2, 4), 0.7))
    assert torch.allclose(frame_levels(state, 0.3, None), torch.full((2, 5), 0.3))


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def set_frame_levels(self, levels):
        self.seen.append(levels.clone())

    def forward(self, noisy, auxiliary):
        return torch.zeros_like(noisy)


def test_the_loss_and_the_sampler_pass_the_frame_levels():
    grid = make_grid(8, 16)
    process = build_process("gaussian", "identity", grid)
    recorder = Recorder()
    levels = torch.rand(2, 5)
    denoising_loss(recorder, process, torch.ones(1), torch.randn(2, 5, 3, 8, 16), torch.randn(2, 5, 2, 8, 16), levels=levels)
    assert torch.equal(recorder.seen[0], levels)
    ddim_sample(ModelDenoiser(recorder), process, torch.zeros(2, 5, 2, 8, 16), 3, 4)
    assert len(recorder.seen) == 5 and torch.allclose(recorder.seen[1], torch.ones(2, 5))


def test_noise_conditioning_makes_the_output_depend_on_the_levels():
    model = randomised(small_model("noise_conditioning"))
    noisy, auxiliary = torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16)
    model.set_frame_levels(torch.zeros(1, 5))
    first = model(noisy, auxiliary)
    model.set_frame_levels(torch.ones(1, 5))
    assert not torch.allclose(first, model(noisy, auxiliary))


def test_degree_filters_add_one_filter_per_channel_and_degree():
    model = randomised(small_model("degree_filters"))
    stage = model.down[0].spectral
    coefficients = torch.randn(2, stage.transform.coefficients, stage.kappa.shape[1])
    degrees = torch.arange(stage.transform.coefficients).float().sqrt().floor().long()
    expected = stage.weight.mT @ coefficients + stage.kappa[degrees] * coefficients
    assert torch.allclose(stage.mix(coefficients), expected, atol=1e-5)


def test_degree_filters_are_restricted_per_channel_on_the_patch():
    source = randomised(small_model("degree_filters"))
    weights = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    regional = transfer_weights(small_model("degree_filters", SMALL_PATCH, SMALL_REGIONAL), weights)
    stage = regional.down[0].spectral
    weight, kappa = weights["down.0.spectral.weight"].double(), weights["down.0.spectral.kappa"].double()
    degrees = torch.arange(weight.shape[0]).float().sqrt().floor().long()
    basis = stage.transform.harmonic_basis(weight.shape[0])
    for channel in (0, kappa.shape[1] - 1):
        expected = restrict(weight + torch.diag(kappa[degrees, channel]), basis)
        assert torch.allclose(stage.restricted[channel].double(), expected, atol=1e-4)
    assert torch.count_nonzero(stage.delta) == 0 and not stage.restricted.requires_grad


def test_space_time_attention_mixes_points_that_temporal_attention_keeps_apart():
    torch.manual_seed(0)
    temporal, joint = TemporalAttention(16, 4, 5), SpaceTimeAttention(16, 4, 5)
    joint.load_state_dict(temporal.state_dict())
    for parameter in [*temporal.parameters(), *joint.parameters()]:
        parameter.data.normal_(0.0, 0.3)
    joint.load_state_dict(temporal.state_dict())
    values = torch.randn(1, 5, 6, 16)
    changed = values.clone()
    changed[:, :, 0] += 1.0
    assert (temporal(changed) - temporal(values))[:, :, 1:].abs().max() < 1e-6
    assert (joint(changed) - joint(values))[:, :, 1:].abs().max() > 1e-3
    model = small_model("space_time_attention")
    assert isinstance(model.middle.attention, SpaceTimeAttention)
    assert not isinstance(model.down[0].attention, SpaceTimeAttention)


@pytest.mark.parametrize("option", ["degree_filters", "noise_conditioning", "space_time_attention"])
def test_every_variant_transfers_to_the_patch(option):
    source = randomised(small_model(option))
    weights = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    regional = apply_adaptation(small_model(option, SMALL_PATCH, SMALL_REGIONAL), "delta_and_pointwise")
    transfer_weights(regional, weights)
    regional.set_frame_levels(torch.rand(1, 5))
    output = regional(torch.randn(1, 5, 6, 8, 16), torch.randn(1, 5, 3, 8, 16))
    assert torch.isfinite(output).all()


def test_the_combined_variant_has_noise_conditioning_and_coefficient_filters():
    spec = load_model_spec("spheridiff_pooled3_noise_coeff")
    assert spec.architecture["coefficient_filters"] and spec.architecture["noise_conditioning"]
    model = build_model(spec.name, spec.architecture, ModelContext(25, 10, 5, make_grid(32, 64), spec.sphere))
    assert count_parameters(model) == 2925310
    assert model.level_embedding is not None
    assert [tuple(block.spectral.kappa.shape) for block in [*model.down, model.middle, *model.up]] == [
        (961, 64), (225, 64), (49, 128), (961, 128), (225, 256)
    ]
    finetuning = load_experiment("finetune_pooled3_noise_coeff")
    assert finetuning.pretrained == "pilot_pooled3_noise_coeff/ema_weights.pt"
    assert load_model_spec(finetuning.model).architecture == spec.architecture


def test_coefficient_filters_add_one_filter_per_channel_and_coefficient():
    model = randomised(small_model("coefficient_filters"))
    stage = model.down[0].spectral
    assert stage.kappa.shape == (stage.transform.coefficients, 8)
    coefficients = torch.randn(2, stage.transform.coefficients, 8)
    expected = stage.weight.mT @ coefficients + stage.kappa * coefficients
    assert torch.allclose(stage.mix(coefficients), expected, atol=1e-5)
    with pytest.raises(ValueError):
        build_model("spheridiff_pooled", {**BASE, "degree_filters": True, "coefficient_filters": True}, ModelContext(6, 3, 5, make_grid(8, 16), SMALL_GLOBAL))


def test_coefficient_filters_are_restricted_per_channel_on_the_patch():
    source = randomised(small_model("coefficient_filters"))
    weights = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    regional = transfer_weights(small_model("coefficient_filters", SMALL_PATCH, SMALL_REGIONAL), weights)
    stage = regional.down[0].spectral
    weight, kappa = weights["down.0.spectral.weight"].double(), weights["down.0.spectral.kappa"].double()
    basis = stage.transform.harmonic_basis(weight.shape[0])
    for channel in (0, kappa.shape[1] - 1):
        expected = restrict(weight + torch.diag(kappa[:, channel]), basis)
        assert torch.allclose(stage.restricted[channel].double(), expected, atol=1e-4)
