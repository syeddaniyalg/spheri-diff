from dataclasses import dataclass

import torch
from torch import nn

from spheridiff.models.layers.init import zeroed
from spheridiff.models.layers.norm import RMSNorm
from spheridiff.models.layers.rope import RotaryEmbedding
from spheridiff.models.wind.blocks import ResBlock, TransformerBlock

from .levels import LevelPlan
from .pooled import SPECTRAL_PRECISIONS
from .spectral import build_mixing


@dataclass(frozen=True)
class HybridConfig:
    base_width: int
    levels: int = 3
    heads: int = 4
    embedding_dim: int = 64
    norm_groups: int = 32
    transformer_blocks: int = 1
    degree_filters: bool = True
    spectral_precision: str = "fp16"

    @property
    def widths(self):
        return [self.base_width * 2**level for level in range(self.levels)]

    def __post_init__(self):
        if self.levels < 2:
            raise ValueError("a hybrid model needs at least two levels")
        if self.spectral_precision not in SPECTRAL_PRECISIONS:
            raise ValueError(f"unknown spectral precision '{self.spectral_precision}', choose from {SPECTRAL_PRECISIONS}")
        for width in self.widths:
            if width % self.norm_groups or width % self.heads:
                raise ValueError(f"width {width} does not suit {self.norm_groups} groups and {self.heads} heads")


class SpectralMixer(nn.Module):
    def __init__(self, transform, width, degree_filters):
        super().__init__()
        self.norm = RMSNorm(width)
        self.spectral = build_mixing(transform, None, width if degree_filters else None)

    def forward(self, values):
        return values + self.spectral(self.norm(values)).to(values.dtype)


class FrameTransformerBlock(TransformerBlock):
    def forward(self, tokens, embedding):
        scale, shift = self.film(embedding).chunk(2, dim=-1)
        normed = self.norm(tokens) * (1 + scale) + shift
        projected, mlp_hidden = self.fused(normed).split((3 * self.dimension, self.mlp_width), dim=-1)
        tokens = tokens + self.attention_out(self.attend(projected))
        return tokens + self.mlp_out(mlp_hidden)


def to_maps(values, height, width):
    batch, frames, points, channels = values.shape
    return values.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)


def from_maps(maps, batch, frames):
    merged, channels, height, width = maps.shape
    return maps.permute(0, 2, 3, 1).reshape(batch, frames, height * width, channels)


class HybridSpherIDiff(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.config, self.frames = config, context.frames
        self.plan = LevelPlan(context.grid, context.sphere, config.levels)
        transforms, widths, deepest = self.plan.transforms, config.widths, config.levels - 1
        self.shapes = [(transform.height, transform.width) for transform in transforms]
        self.mixers = nn.ModuleList(
            [SpectralMixer(transforms[level], widths[level], config.degree_filters) for level in range(config.levels)]
        )
        self.uplift = nn.Linear(context.channels + context.auxiliary_channels, widths[0])
        self.down_blocks = nn.ModuleList([self.res_block(widths[level]) for level in range(deepest)])
        self.up_blocks = nn.ModuleList([self.res_block(widths[level]) for level in range(deepest)])
        self.widen = nn.ModuleList([nn.Linear(widths[level], widths[level + 1]) for level in range(deepest)])
        self.narrow = nn.ModuleList([nn.Linear(widths[level + 1], widths[level]) for level in range(deepest)])
        head_dimension = widths[deepest] // config.heads
        self.rotary = RotaryEmbedding(head_dimension, (context.frames, *self.shapes[deepest]))
        self.transformers = nn.ModuleList(
            [FrameTransformerBlock(widths[deepest], config.heads, config.embedding_dim, self.rotary) for _ in range(config.transformer_blocks)]
        )
        self.level_embedding = nn.Sequential(
            nn.Linear(1, config.embedding_dim), nn.SiLU(), nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        self.downlift = zeroed(nn.Linear(widths[0], context.channels))
        self.frame_levels = None
        self.set_half_precision(config.spectral_precision == "fp16")

    def res_block(self, width):
        return ResBlock(width, self.config.embedding_dim, self.config.norm_groups)

    def set_half_precision(self, enabled):
        self.plan.set_half_precision(enabled)
        for module in self.modules():
            if hasattr(module, "mix"):
                module.half_precision = enabled

    def set_frame_levels(self, levels):
        self.frame_levels = levels

    def embedding(self, batch, frames, device, dtype):
        levels = self.frame_levels
        if levels is None or levels.shape != (batch, frames):
            levels = torch.zeros(batch, frames, device=device)
        return self.level_embedding(levels.to(device=device, dtype=dtype)[..., None])

    def local(self, block, values, level, embedding):
        batch, frames = values.shape[:2]
        maps = block(to_maps(values, *self.shapes[level]), embedding.reshape(batch * frames, -1).to(values.dtype))
        return from_maps(maps, batch, frames)

    def attend(self, values, embedding):
        batch, frames, points, channels = values.shape
        tokens = values.reshape(batch, frames * points, channels)
        conditioning = embedding[:, :, None, :].expand(batch, frames, points, -1).reshape(batch, frames * points, -1)
        for block in self.transformers:
            tokens = block(tokens, conditioning.to(tokens.dtype))
        return tokens.reshape(batch, frames, points, channels)

    def forward(self, noisy, auxiliary):
        batch, frames, _, height, width = noisy.shape
        merged = torch.cat([noisy, auxiliary], dim=2)
        values = self.uplift(merged.permute(0, 1, 3, 4, 2).reshape(batch, frames, height * width, -1))
        embedding = self.embedding(batch, frames, values.device, values.dtype)
        skips = []
        for level, block in enumerate(self.down_blocks):
            values = self.local(block, self.mixers[level](values), level, embedding)
            skips.append(values)
            values = self.widen[level](self.plan.pools[level](values))
        values = self.attend(self.mixers[-1](values), embedding)
        for level in reversed(range(len(self.up_blocks))):
            values = self.plan.unpools[level](self.narrow[level](values)) + skips[level]
            values = self.local(self.up_blocks[level], self.mixers[level](values), level, embedding)
        output = self.downlift(values).reshape(batch, frames, height, width, -1)
        return output.permute(0, 1, 4, 2, 3)


def build_hybrid(architecture, context):
    converted = {key: tuple(value) if isinstance(value, list) else value for key, value in architecture.items()}
    return HybridSpherIDiff(HybridConfig(**converted), context)
