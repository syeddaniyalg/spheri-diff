import torch
from torch import nn

from spheridiff.models.layers.init import zeroed
from spheridiff.sphere.transform import SphericalTransform

from .blocks import SpherIDiffBlock


class SpherIDiff(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        transform = SphericalTransform(context.grid)
        widths = config.widths
        self.uplift = nn.Linear(context.channels + context.auxiliary_channels, widths[0])
        self.blocks = nn.ModuleList(
            [
                SpherIDiffBlock(widths[i], widths[i + 1], config.heads, context.frames, transform)
                for i in range(config.blocks)
            ]
        )
        self.downlift = zeroed(nn.Linear(widths[-1], context.channels))

    def forward(self, noisy, auxiliary):
        batch, frames, _, height, width = noisy.shape
        merged = torch.cat([noisy, auxiliary], dim=2)
        points = merged.permute(0, 1, 3, 4, 2).reshape(batch, frames, height * width, -1)
        values = self.uplift(points)
        for block in self.blocks:
            values = block(values)
        output = self.downlift(values).reshape(batch, frames, height, width, -1)
        return output.permute(0, 1, 4, 2, 3)
