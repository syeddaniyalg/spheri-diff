import torch
from torch import nn

from spheridiff.grid.grid import gauss_legendre_grid
from spheridiff.sphere.separable import SeparableTransform
from spheridiff.sphere.transform import SphericalTransform, build_transform


class Resampler(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        matrix = source.analysis.double() @ target.synthesis.double()
        self.register_buffer("matrix", matrix.float(), persistent=False)

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            return torch.matmul(self.matrix.mT, values.float())


class GridPlan(nn.Module):
    def __init__(self, block_transform, incoming=None, outgoing=None):
        super().__init__()
        self.block_transform = block_transform
        self.incoming = incoming
        self.outgoing = outgoing

    def enter(self, values):
        return values if self.incoming is None else self.incoming(values).to(values.dtype)

    def leave(self, values):
        return values if self.outgoing is None else self.outgoing(values).to(values.dtype)


def internal_transform(grid, degrees, kind):
    if kind == "separable":
        return SeparableTransform(grid, degrees)
    return SphericalTransform(grid, degrees)


def build_grid_plan(config, context):
    data_transform = build_transform(context.grid, context.sphere)
    if config.internal_grid == "data":
        return GridPlan(data_transform)
    degrees = data_transform.degrees
    internal = gauss_legendre_grid(context.grid.height, context.grid.width)
    dense_internal = SphericalTransform(internal, degrees)
    incoming = Resampler(data_transform, dense_internal)
    outgoing = Resampler(dense_internal, data_transform)
    return GridPlan(internal_transform(internal, degrees, config.transform), incoming, outgoing)
