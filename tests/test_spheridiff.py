import pytest
import torch
import torch.nn.functional as functional

from helpers import make_grid, randomise_weights
from spheridiff.models.context import ModelContext
from spheridiff.models.layers.norm import RMSNorm
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.spheridiff.blocks import TemporalAttention
from spheridiff.models.spheridiff.config import SpherIDiffConfig
from spheridiff.models.spheridiff.spectral import SpectralMixing
from spheridiff.sphere.transform import SphericalTransform

WIDTHS = [16, 32, 32, 32, 16]


def make_context(channels=6, auxiliary=3, frames=5):
    return ModelContext(channels, auxiliary, frames, make_grid(16, 32))


def make_model(widths=WIDTHS, heads=4, context=None):
    return build_model("spheridiff", {"widths": widths, "heads": heads}, context or make_context())


def expected_parameters(widths, coefficients, channels, auxiliary):
    total = (channels + auxiliary) * widths[0] + widths[0]
    for incoming, outgoing in zip(widths[:-1], widths[1:]):
        total += 2 * incoming + outgoing + coefficients**2
        total += incoming * outgoing + outgoing + outgoing * outgoing + outgoing
        total += incoming * outgoing if incoming != outgoing else 0
        total += 4 * outgoing**2
    return total + widths[-1] * channels + channels


def test_forward_shape_and_zero_start():
    model = make_model()
    noisy = torch.randn(2, 5, 6, 16, 32)
    auxiliary = torch.randn(2, 5, 3, 16, 32)
    output = model(noisy, auxiliary)
    assert output.shape == noisy.shape
    assert torch.count_nonzero(output) == 0


def test_parameter_count_matches_the_formula():
    context = make_context()
    model = make_model(context=context)
    coefficients = context.grid.coefficients
    assert count_parameters(model) == expected_parameters(WIDTHS, coefficients, 6, 3)


def test_pilot_sized_model_stays_below_the_smallest_wind():
    context = ModelContext(25, 10, 5, make_grid(32, 64))
    with torch.device("meta"):
        model = build_model(
            "spheridiff", {"widths": [128, 256, 256, 256, 128], "heads": 4}, context
        )
    assert 1.4e6 < count_parameters(model) < 1.7e6


def test_all_parameters_learn():
    torch.manual_seed(0)
    model = make_model()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    noisy = torch.randn(2, 5, 6, 16, 32)
    auxiliary = torch.randn(2, 5, 3, 16, 32)
    target = torch.randn(2, 5, 6, 16, 32)
    for _ in range(10):
        optimizer.zero_grad()
        ((model(noisy, auxiliary) - target) ** 2).mean().backward()
        optimizer.step()
    unchanged = [
        name
        for (name, parameter), start in zip(model.named_parameters(), initial)
        if torch.equal(parameter, start)
    ]
    assert unchanged == []


def test_identity_spectral_weight_band_limits_the_field():
    grid = make_grid(32, 64)
    transform = SphericalTransform(grid)
    mixing = SpectralMixing(transform)
    mixing.weight.data.copy_(torch.eye(transform.coefficients))
    field = torch.randn(2, 5, 4, 32, 64)
    points = field.permute(0, 1, 3, 4, 2).reshape(2, 5, 32 * 64, 4)
    mixed = mixing(points).reshape(2, 5, 32, 64, 4).permute(0, 1, 4, 2, 3)
    assert torch.allclose(mixed, transform.band_limit(field), atol=1e-4)


def test_spectral_mixing_shares_weights_across_channels_and_frames():
    transform = SphericalTransform(make_grid(16, 32))
    mixing = SpectralMixing(transform)
    mixing.weight.data.normal_()
    values = torch.randn(1, 5, 16 * 32, 4)
    changed = values.clone()
    changed[:, :, :, 0] += 1.0
    difference = mixing(changed) - mixing(values)
    assert difference[..., 1:].abs().max() < 1e-4
    assert difference[..., 0].abs().max() > 1e-3


def test_model_is_sensitive_to_frame_order():
    torch.manual_seed(0)
    model = randomise_weights(make_model(), 0.3)
    noisy = torch.randn(1, 5, 6, 16, 32)
    auxiliary = torch.randn(1, 5, 3, 16, 32)
    order = [4, 3, 2, 1, 0]
    forward = model(noisy, auxiliary)
    reverse = model(noisy[:, order], auxiliary[:, order])
    assert not torch.allclose(forward[:, order], reverse, atol=1e-4)


def reference_attention(module, values):
    batch, frames, points, width = values.shape
    sequences = values.permute(0, 2, 1, 3).reshape(batch * points, frames, width)
    shaped = module.projection(sequences).view(batch * points, frames, 3, module.heads, -1)
    query, key, value = shaped.permute(2, 0, 3, 1, 4).unbind(0)
    attended = functional.scaled_dot_product_attention(module.rotary(query), module.rotary(key), value)
    merged = attended.transpose(1, 2).reshape(batch * points, frames, width)
    return module.output(merged).reshape(batch, points, frames, width).permute(0, 2, 1, 3)


def test_temporal_attention_matches_the_scaled_dot_product_reference():
    torch.manual_seed(0)
    module = randomise_weights(TemporalAttention(32, 4, 5), 0.3)
    values = torch.randn(2, 5, 7, 32)
    assert torch.allclose(module(values), reference_attention(module, values), atol=1e-5)


def test_temporal_attention_gradients_match_the_reference():
    torch.manual_seed(0)
    module = randomise_weights(TemporalAttention(32, 4, 5), 0.3)
    values = torch.randn(2, 5, 7, 32, requires_grad=True)
    module(values).square().sum().backward()
    fast = values.grad.clone()
    values.grad = None
    reference_attention(module, values).square().sum().backward()
    assert torch.allclose(fast, values.grad, atol=1e-4)


def test_fused_norm_matches_the_reference_formula():
    torch.manual_seed(0)
    norm = RMSNorm(24)
    norm.weight.data.uniform_(0.5, 1.5)
    values = torch.randn(3, 5, 24) * 4.0
    assert torch.allclose(norm.fused(values), norm.reference(values), atol=1e-6)
    assert torch.allclose(norm(values), norm.reference(values), atol=1e-6)


def test_fused_norm_keeps_low_precision_close_to_the_reference():
    norm = RMSNorm(24)
    values = (torch.randn(3, 5, 24) * 4.0).to(torch.bfloat16)
    fast = norm.fused(values).float()
    exact = norm.reference(values.float())
    assert ((fast - exact).norm() / exact.norm()).item() < 1e-2


def test_invalid_heads_are_rejected():
    with pytest.raises(ValueError):
        SpherIDiffConfig(widths=(16, 30, 16), heads=4)
    with pytest.raises(ValueError):
        SpherIDiffConfig(widths=(16,))
