import torch
import torch.nn.functional as functional
from torch import nn

from spheridiff.models.layers.init import zeroed
from spheridiff.models.layers.rope import RotaryEmbedding

from .blocks import ResBlock, TransformerBlock
from .resample import Downsample, Upsample


def to_tokens(values, frames):
    merged, channels, height, width = values.shape
    shaped = values.reshape(merged // frames, frames, channels, height, width)
    return shaped.permute(0, 1, 3, 4, 2).reshape(merged // frames, frames * height * width, channels)


def from_tokens(tokens, frames, height, width):
    batch, _, channels = tokens.shape
    shaped = tokens.reshape(batch, frames, height, width, channels)
    return shaped.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)


def resize(values, size):
    if tuple(values.shape[-2:]) == tuple(size):
        return values
    return functional.interpolate(values, size=size, mode="bilinear", align_corners=False)


class Wind(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.config = config
        self.frames = context.frames
        self.model_size = (
            config.model_height or context.grid.height,
            config.model_width or context.grid.width,
        )
        self.check_sizes()
        patch = config.patch_size
        self.embed_input = nn.Conv2d(
            context.channels + context.auxiliary_channels, config.widths[0], patch, stride=patch
        )
        self.project_output = zeroed(
            nn.ConvTranspose2d(config.widths[0], context.channels, patch, stride=patch)
        )
        self.noise_embedding = nn.Parameter(torch.randn(1, config.embedding_dim))
        self.rotaries = self.build_rotaries()
        self.build_stages()

    def level_size(self, level):
        patch = self.config.patch_size
        return self.model_size[0] // patch // 2**level, self.model_size[1] // patch // 2**level

    def check_sizes(self):
        patch, depth = self.config.patch_size, self.config.levels - 1
        height, width = self.model_size
        if height % (patch * 2**depth) or width % (patch * 2**depth):
            raise ValueError(f"the model size {self.model_size} does not suit the patch size and levels")

    def build_rotaries(self):
        rotaries = nn.ModuleDict()
        for level, kind in enumerate(self.config.block_kinds):
            if kind == "transformer":
                head_dimension = self.config.widths[level] // self.config.heads
                sizes = (self.frames, *self.level_size(level))
                rotaries[str(level)] = RotaryEmbedding(head_dimension, sizes)
        return rotaries

    def make_block(self, level):
        width, kind = self.config.widths[level], self.config.block_kinds[level]
        if kind == "res":
            return ResBlock(width, self.config.embedding_dim, self.config.norm_groups)
        rotary = self.rotaries[str(level)]
        return TransformerBlock(width, self.config.heads, self.config.embedding_dim, rotary)

    def make_stage(self, level, count):
        return nn.ModuleList([self.make_block(level) for _ in range(count)])

    def build_stages(self):
        widths, counts = self.config.widths, self.config.blocks_per_level
        depth = self.config.levels - 1
        self.down_stages = nn.ModuleList([self.make_stage(i, counts[i]) for i in range(depth)])
        self.downsamplers = nn.ModuleList([Downsample(widths[i], widths[i + 1]) for i in range(depth)])
        self.middle = self.make_stage(depth, self.config.middle_blocks)
        self.upsamplers = nn.ModuleList([Upsample(widths[i + 1], widths[i]) for i in range(depth)])
        self.up_stages = nn.ModuleList([self.make_stage(i, counts[i]) for i in range(depth)])

    def run_stage(self, stage, values, level, embedding):
        if self.config.block_kinds[level] == "res":
            for block in stage:
                values = block(values, embedding)
            return values
        height, width = values.shape[-2:]
        tokens = to_tokens(values, self.frames)
        for block in stage:
            tokens = block(tokens, embedding)
        return from_tokens(tokens, self.frames, height, width)

    def encode(self, values, embedding):
        before, after = [], []
        for level, stage in enumerate(self.down_stages):
            values = self.run_stage(stage, values, level, embedding)
            before.append(values)
            values = self.downsamplers[level](values)
            after.append(values)
        return values, before, after

    def decode(self, values, before, after, embedding):
        for level in reversed(range(self.config.levels - 1)):
            values = values - after.pop()
            values = self.upsamplers[level](values) + before.pop()
            values = self.run_stage(self.up_stages[level], values, level, embedding)
        return values

    def forward(self, noisy, auxiliary):
        batch, frames = noisy.shape[:2]
        original_size = tuple(noisy.shape[-2:])
        merged = torch.cat([noisy, auxiliary], dim=2).flatten(0, 1)
        values = self.embed_input(resize(merged, self.model_size))
        embedding = self.noise_embedding
        values, before, after = self.encode(values, embedding)
        values = self.run_stage(self.middle, values, self.config.levels - 1, embedding)
        values = self.decode(values, before, after, embedding)
        output = resize(self.project_output(values), original_size)
        return output.unflatten(0, (batch, frames))
