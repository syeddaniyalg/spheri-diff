import torch
from torch import nn

from spheridiff.models.layers.init import zeroed

from .blocks import SpherIDiffBlock
from .grids import build_grid_plan


class SpherIDiff(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        self.grids = build_grid_plan(config, context)
        transform = self.grids.block_transform
        widths = config.widths
        self.uplift = nn.Linear(context.channels + context.auxiliary_channels, widths[0])
        self.blocks = nn.ModuleList(
            [
                SpherIDiffBlock(widths[i], widths[i + 1], config.heads, context.frames, transform, config.rank)
                for i in range(config.blocks)
            ]
        )
        self.downlift = zeroed(nn.Linear(widths[-1], context.channels))

    def forward(self, noisy, auxiliary):
        batch, frames, _, height, width = noisy.shape
        merged = torch.cat([noisy, auxiliary], dim=2)
        points = merged.permute(0, 1, 3, 4, 2).reshape(batch, frames, height * width, -1)
        values = self.uplift(self.grids.enter(points))
        for block in self.blocks:
            values = block(values)
        output = self.grids.leave(self.downlift(values)).reshape(batch, frames, height, width, -1)
        return output.permute(0, 1, 4, 2, 3)
