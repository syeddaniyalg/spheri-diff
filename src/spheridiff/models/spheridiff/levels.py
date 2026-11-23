import numpy as np
import torch
from torch import nn

from spheridiff.grid.grid import GridSpec
from spheridiff.sphere.slepian import KernelSlepianTransform, cell_bounds, least_squares, patch_weights
from spheridiff.sphere.transform import SphericalTransform


class Resample(nn.Module):
    half_precision = False

    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix, dtype=torch.float32), persistent=False)

    def forward(self, values):
        if self.half_precision:
            return torch.matmul(self.matrix.mT, values)
        with torch.autocast(device_type=values.device.type, enabled=False):
            return torch.matmul(self.matrix.mT, values.float()).to(values.dtype)


def least_squares_limit(grid):
    return min(grid.height - 1, grid.width // 2)


def block_centres(values, factor):
    return np.asarray(values, dtype=np.float64).reshape(-1, factor).mean(axis=1)


def check_divisible(grid, levels):
    factor = 2 ** (levels - 1)
    if grid.height % factor or grid.width % factor:
        raise ValueError(f"a {grid.height} x {grid.width} grid cannot be pooled {levels - 1} times by 2")


def global_grids(grid, levels):
    grids = [grid]
    for level in range(1, levels):
        height, width = grid.height // 2**level, grid.width // 2**level
        grids.append(GridSpec(np.linspace(-90.0, 90.0, height), np.arange(width) * 360.0 / width))
    return grids


def regional_grids(grid, levels):
    grids = [grid]
    for level in range(1, levels):
        factor = 2**level
        grids.append(GridSpec(block_centres(grid.latitudes, factor), block_centres(grid.longitudes, factor)))
    return grids


def global_transforms(grids, sphere):
    top = sphere.get("bandlimit") or least_squares_limit(grids[0])
    limits = [min(top, least_squares_limit(grid)) if level else top for level, grid in enumerate(grids)]
    return [SphericalTransform(grid, limit, "least_squares") for grid, limit in zip(grids, limits)]


def regional_transforms(grids, sphere):
    bounds = (cell_bounds(grids[0].latitudes), cell_bounds(grids[0].longitudes))
    top = sphere["bandlimit"]
    return [KernelSlepianTransform(grid, top // 2**level, bounds) for level, grid in enumerate(grids)]


def truncation_pair(fine, coarse):
    kept = coarse.coefficients
    down = fine.analysis[:, :kept].double() @ coarse.synthesis.double()
    up = coarse.analysis.double() @ fine.synthesis[:kept].double()
    return Resample(down), Resample(up)


def projection_pair(fine_grid, coarse):
    functions = coarse.functions_at(fine_grid.latitudes, fine_grid.longitudes)
    fit = least_squares(functions, patch_weights(fine_grid))
    down = torch.as_tensor(fit) @ coarse.synthesis.double()
    up = coarse.analysis.double() @ torch.as_tensor(functions.T)
    return Resample(down), Resample(up)


class LevelPlan(nn.Module):
    def __init__(self, grid, sphere, levels):
        super().__init__()
        check_divisible(grid, levels)
        regional = sphere.get("kind") == "slepian"
        grids = regional_grids(grid, levels) if regional else global_grids(grid, levels)
        transforms = regional_transforms(grids, sphere) if regional else global_transforms(grids, sphere)
        self.transforms = nn.ModuleList(transforms)
        pairs = [
            projection_pair(grids[level], transforms[level + 1]) if regional else truncation_pair(transforms[level], transforms[level + 1])
            for level in range(levels - 1)
        ]
        self.pools = nn.ModuleList([down for down, _ in pairs])
        self.unpools = nn.ModuleList([up for _, up in pairs])

    def set_half_precision(self, enabled):
        for resample in [*self.pools, *self.unpools]:
            resample.half_precision = enabled
