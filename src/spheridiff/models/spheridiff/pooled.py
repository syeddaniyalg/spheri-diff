from dataclasses import dataclass

import torch
from torch import nn

from spheridiff.models.layers.init import zeroed

from .blocks import SpaceTimeAttention, SpherIDiffBlock
from .levels import LevelPlan
from .local import LocalStage

SPECTRAL_PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class PooledConfig:
    base_width: int
    levels: int
    heads: int = 4
    rank: int = None
    spectral_precision: str = "fp32"
    degree_filters: bool = False
    coefficient_filters: bool = False
    noise_conditioning: bool = False
    space_time_attention: bool = False
    spherical_convolution: bool = False
    channel_knots: int = None
    pretrained_degrees: tuple = None

    @property
    def widths(self):
        return [self.base_width * 2**level for level in range(self.levels)]

    def pretrained_degree(self, level):
        return None if self.pretrained_degrees is None else self.pretrained_degrees[level]

    def check_channel_mixing(self):
        if self.channel_knots is None:
            return
        if self.rank is not None or self.degree_filters or self.coefficient_filters:
            raise ValueError("channel mixing replaces the mode-mixing weight, its rank and its filters")
        if self.pretrained_degrees is not None and len(self.pretrained_degrees) != self.levels:
            raise ValueError("give one pretrained band limit per level")

    def __post_init__(self):
        if self.levels < 2:
            raise ValueError("a pooled model needs at least two levels")
        self.check_channel_mixing()
        if self.degree_filters and self.coefficient_filters:
            raise ValueError("choose either degree filters or coefficient filters, not both")
        if self.spectral_precision not in SPECTRAL_PRECISIONS:
            raise ValueError(f"unknown spectral precision '{self.spectral_precision}', choose from {SPECTRAL_PRECISIONS}")
        for width in self.widths:
            if width % self.heads or (width // self.heads) % 2:
                raise ValueError(f"width {width} does not suit {self.heads} attention heads")


class PooledSpherIDiff(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.plan = LevelPlan(context.grid, context.sphere, config.levels)
        transforms, widths, frames = self.plan.transforms, config.widths, context.frames

        def block(level, incoming, outgoing):
            return SpherIDiffBlock(
                incoming,
                outgoing,
                config.heads,
                frames,
                transforms[level],
                config.rank,
                config.degree_filters,
                config.coefficient_filters,
                config.channel_knots,
                config.pretrained_degree(level),
            )

        deepest = config.levels - 1
        self.uplift = nn.Linear(context.channels + context.auxiliary_channels, widths[0])
        self.down = nn.ModuleList([block(level, widths[max(level - 1, 0)], widths[level]) for level in range(deepest)])
        self.middle = block(deepest, widths[deepest - 1], widths[deepest])
        self.up = nn.ModuleList([block(level, widths[level + 1], widths[level]) for level in range(deepest)])
        self.downlift = zeroed(nn.Linear(widths[0], context.channels))
        if config.space_time_attention:
            self.middle.attention = SpaceTimeAttention(widths[deepest], config.heads, frames)
        if config.spherical_convolution:
            regional = context.sphere.get("kind") == "slepian"
            for top in (self.down[0], self.up[0]):
                top.local = LocalStage(context.grid, widths[0], regional)
        self.level_embedding = None
        if config.noise_conditioning:
            self.level_embedding = nn.Sequential(nn.Linear(1, widths[0]), nn.SiLU(), zeroed(nn.Linear(widths[0], widths[0])))
        self.frame_levels = None
        self.set_half_precision(config.spectral_precision == "fp16")

    def set_frame_levels(self, levels):
        self.frame_levels = levels

    def condition(self, values):
        if self.level_embedding is None:
            return values
        batch, frames = values.shape[:2]
        levels = self.frame_levels
        if levels is None or levels.shape != (batch, frames):
            levels = torch.zeros(batch, frames, device=values.device)
        embedding = self.level_embedding(levels.to(device=values.device, dtype=values.dtype)[..., None])
        return values + embedding[:, :, None, :].to(values.dtype)

    def set_half_precision(self, enabled):
        self.plan.set_half_precision(enabled)
        for module in self.modules():
            if hasattr(module, "mix"):
                module.half_precision = enabled

    def forward(self, noisy, auxiliary):
        batch, frames, _, height, width = noisy.shape
        merged = torch.cat([noisy, auxiliary], dim=2)
        values = self.condition(self.uplift(merged.permute(0, 1, 3, 4, 2).reshape(batch, frames, height * width, -1)))
        skips = []
        for level, block in enumerate(self.down):
            values = block(values if level == 0 else self.plan.pools[level - 1](values))
            skips.append(values)
        values = self.middle(self.plan.pools[-1](values))
        for level in reversed(range(len(self.up))):
            values = self.up[level](self.plan.unpools[level](values)) + skips[level]
        output = self.downlift(values).reshape(batch, frames, height, width, -1)
        return output.permute(0, 1, 4, 2, 3)


def build_pooled(architecture, context):
    converted = {key: tuple(value) if isinstance(value, list) else value for key, value in architecture.items()}
    return PooledSpherIDiff(PooledConfig(**converted), context)
