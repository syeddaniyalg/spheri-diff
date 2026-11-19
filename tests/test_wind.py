import pytest
import torch

from helpers import make_grid, randomise_weights
from spheridiff.models.context import ModelContext
from spheridiff.models.layers.rope import RotaryEmbedding, axis_feature_counts
from spheridiff.models.registry import build_model, count_parameters
from spheridiff.models.wind.config import WindConfig

SMALL = dict(
    widths=(64, 128, 128, 128),
    block_kinds=("res", "res", "transformer", "transformer"),
    blocks_per_level=(1, 1, 1),
    middle_blocks=1,
)


def make_context(height=32, width=64, channels=6, auxiliary=3, frames=5):
    return ModelContext(channels, auxiliary, frames, make_grid(height, width))


def make_wind(context=None, **overrides):
    architecture = {**SMALL, **{key: list(value) if isinstance(value, tuple) else value for key, value in overrides.items()}}
    architecture = {key: list(value) if isinstance(value, tuple) else value for key, value in architecture.items()}
    return build_model("wind", architecture, context or make_context())


def residual_parameters(channels, embedding):
    return 18 * channels**2 + 2 * embedding * channels + 8 * channels


def transformer_parameters(dimension, heads, embedding):
    head = dimension // heads
    film = embedding * 2 * dimension + 2 * dimension
    return film + dimension + 7 * dimension**2 + 7 * dimension + 2 * head + 5 * dimension**2 + 2 * dimension


def expected_parameters(config, channels, auxiliary):
    widths, embedding, patch = config.widths, config.embedding_dim, config.patch_size
    total = (channels + auxiliary) * widths[0] * patch**2 + widths[0]
    total += widths[0] * channels * patch**2 + channels + embedding
    for level in range(config.levels):
        count = config.blocks_per_level[level] * 2 if level < config.levels - 1 else config.middle_blocks
        if config.block_kinds[level] == "res":
            total += count * residual_parameters(widths[level], embedding)
        else:
            total += count * transformer_parameters(widths[level], config.heads, embedding)
    for upper, lower in zip(widths[:-1], widths[1:]):
        total += 9 * upper * (lower // 4) + lower // 4
        total += 9 * lower * upper * 4 + upper * 4
    return total


def test_forward_shape_and_zero_start():
    context = make_context()
    model = make_wind(context)
    noisy = torch.randn(2, 5, 6, 32, 64)
    auxiliary = torch.randn(2, 5, 3, 32, 64)
    output = model(noisy, auxiliary)
    assert output.shape == noisy.shape
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("patch_size", [1, 2])
def test_parameter_count_matches_the_formula(patch_size):
    context = make_context()
    model = make_wind(context, patch_size=patch_size)
    config = WindConfig(**{**SMALL, "patch_size": patch_size})
    assert count_parameters(model) == expected_parameters(config, 6, 3)


def test_full_size_model_matches_the_published_parameter_count():
    architecture = dict(
        widths=[256, 512, 1024, 2048],
        block_kinds=["res", "res", "transformer", "transformer"],
        blocks_per_level=[4, 4, 4],
        middle_blocks=4,
        patch_size=2,
        model_height=128,
        model_width=240,
    )
    context = ModelContext(70, 10, 5, make_grid(121, 240))
    with torch.device("meta"):
        model = build_model("wind", architecture, context)
    assert 457.0e6 < count_parameters(model) < 458.5e6


def test_all_parameters_learn():
    torch.manual_seed(0)
    model = make_wind()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    noisy = torch.randn(2, 5, 6, 32, 64)
    auxiliary = torch.randn(2, 5, 3, 32, 64)
    target = torch.randn(2, 5, 6, 32, 64)
    for _ in range(8):
        optimizer.zero_grad()
        ((model(noisy, auxiliary) - target) ** 2).mean().backward()
        optimizer.step()
    unchanged = [
        name
        for (name, parameter), start in zip(model.named_parameters(), initial)
        if torch.equal(parameter, start)
    ]
    assert unchanged == []


def test_model_is_sensitive_to_frame_order():
    torch.manual_seed(0)
    model = randomise_weights(make_wind(), 0.15)
    noisy = torch.randn(1, 5, 6, 32, 64)
    auxiliary = torch.randn(1, 5, 3, 32, 64)
    order = [4, 3, 2, 1, 0]
    forward = model(noisy, auxiliary)
    reverse = model(noisy[:, order], auxiliary[:, order])
    assert not torch.allclose(forward[:, order], reverse, atol=1e-4)


def test_resizing_path_returns_the_input_size():
    context = make_context(height=12, width=24)
    model = make_wind(context, model_height=16, model_width=32)
    output = model(torch.randn(1, 5, 6, 12, 24), torch.randn(1, 5, 3, 12, 24))
    assert output.shape == (1, 5, 6, 12, 24)


def test_incompatible_widths_are_rejected():
    with pytest.raises(ValueError):
        WindConfig(**{**SMALL, "widths": (64, 96, 128, 192)})
    with pytest.raises(ValueError):
        WindConfig(**{**SMALL, "widths": (48, 96, 96, 96)})
    with pytest.raises(ValueError):
        WindConfig(**{**SMALL, "block_kinds": ("res", "res", "transformer")})


def test_unsuitable_grid_is_rejected():
    with pytest.raises(ValueError):
        make_wind(make_context(height=30, width=64))


def test_rotary_axis_split_matches_the_reference_rule():
    assert axis_feature_counts(32, 3) == [12, 10, 10]
    assert axis_feature_counts(64, 3) == [20, 22, 22]
    assert axis_feature_counts(48, 3) == [16, 16, 16]
    assert axis_feature_counts(32, 1) == [32]


def test_rotary_scores_depend_on_relative_position():
    rotary = RotaryEmbedding(16, (2, 3, 4))
    vector = torch.randn(16)
    queries = vector.expand(1, 1, 24, 16).clone()
    rotated = rotary(queries)[0, 0]
    scores = rotated @ rotated.T
    first = scores[1 * 4 + 1, 1 * 4 + 3]
    second = scores[1 * 4 + 0, 1 * 4 + 2]
    assert torch.allclose(first, second, atol=1e-5)
    other = scores[0 * 4 + 1, 1 * 4 + 1]
    assert not torch.allclose(first, other, atol=1e-5)
