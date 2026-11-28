import numpy as np
import torch
from torch import nn


class SpectralStage(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    half_precision = False

    def forward(self, values):
        if self.half_precision:
            return self.transform.synthesise_points(self.mix(self.transform.analyse_points(values)))
        with torch.autocast(device_type=values.device.type, enabled=False):
            coefficients = self.transform.analyse_points(values.float())
            return self.transform.synthesise_points(self.mix(coefficients))


class SpectralMixing(SpectralStage):
    def __init__(self, transform):
        super().__init__(transform)
        size = transform.coefficients
        self.weight = nn.Parameter(torch.zeros(size, size))

    def mix(self, coefficients):
        return torch.matmul(self.weight.mT, coefficients)

    def dense_weight(self):
        return self.weight.detach().clone()


class DiagonalLowRankMixing(SpectralStage):
    def __init__(self, transform, rank):
        super().__init__(transform)
        size = transform.coefficients
        self.diagonal = nn.Parameter(torch.zeros(size))
        self.left = nn.Parameter(torch.randn(size, rank) / size**0.5)
        self.right = nn.Parameter(torch.zeros(size, rank))

    def mix(self, coefficients):
        diagonal = self.diagonal[:, None] * coefficients
        return diagonal + torch.matmul(self.right, torch.matmul(self.left.mT, coefficients))

    def dense_weight(self):
        with torch.no_grad():
            return torch.diag(self.diagonal) + self.left @ self.right.mT


class SlepianMixing(SpectralStage):
    def __init__(self, transform):
        super().__init__(transform)
        size = transform.coefficients
        self.restricted = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self.delta = nn.Parameter(torch.zeros(size, size))

    def mix(self, coefficients):
        return torch.matmul((self.restricted + self.delta).mT, coefficients)

    def dense_weight(self):
        with torch.no_grad():
            return self.restricted + self.delta


def degree_index(coefficients):
    return torch.arange(coefficients).float().sqrt().floor().long()


class DegreeFilteredMixing(SpectralMixing):
    def __init__(self, transform, width):
        super().__init__(transform)
        self.kappa = nn.Parameter(torch.zeros(transform.degrees, width))
        self.register_buffer("degrees_of", degree_index(transform.coefficients), persistent=False)

    def mix(self, coefficients):
        return super().mix(coefficients) + self.kappa[self.degrees_of] * coefficients


class CoefficientFilteredMixing(SpectralMixing):
    def __init__(self, transform, width):
        super().__init__(transform)
        self.kappa = nn.Parameter(torch.zeros(transform.coefficients, width))

    def mix(self, coefficients):
        return super().mix(coefficients) + self.kappa * coefficients


class SlepianChannelMixing(SpectralStage):
    def __init__(self, transform, width):
        super().__init__(transform)
        size = transform.coefficients
        self.restricted = nn.Parameter(torch.zeros(width, size, size), requires_grad=False)
        self.delta = nn.Parameter(torch.zeros(size, size))

    def mix(self, coefficients):
        own = torch.einsum("cij,...ic->...jc", self.restricted.to(coefficients.dtype), coefficients)
        return own + torch.matmul(self.delta.mT, coefficients)


def knot_positions(degrees, knots):
    return np.linspace(0.0, degrees - 1.0, min(knots, degrees))


def hat_basis(positions, degrees):
    identity = np.eye(len(positions))
    return np.stack([np.interp(np.arange(degrees), positions, identity[k]) for k in range(len(positions))], axis=1)


def extended_positions(pretrained, pretrained_degrees, degrees):
    added = np.linspace(float(pretrained_degrees), degrees - 1.0, min(len(pretrained), degrees - pretrained_degrees))
    return np.concatenate([pretrained, added])


def regional_basis(degrees, knots, pretrained_degrees):
    if degrees <= pretrained_degrees:
        raise ValueError(f"the regional band limit {degrees} must exceed the pretrained one {pretrained_degrees}")
    pretrained = knot_positions(pretrained_degrees, knots)
    basis = hat_basis(extended_positions(pretrained, pretrained_degrees, degrees), degrees)
    basis[pretrained_degrees:, : len(pretrained)] = 0.0
    return basis, len(pretrained)


def signed_orders(degrees):
    return np.concatenate([np.arange(-degree, degree + 1) for degree in range(degrees)])


def partner_index(degrees):
    index = np.arange(degrees * degrees)
    degree = np.floor(np.sqrt(index)).astype(int)
    return degree * degree + degree - signed_orders(degrees)


class DegreeChannelMixing(SpectralStage):
    def __init__(self, transform, width, knots):
        super().__init__(transform)
        degrees = transform.degrees
        basis = hat_basis(knot_positions(degrees, knots), degrees)
        self.isotropic = nn.Parameter(torch.zeros(basis.shape[1], width, width))
        self.zonal = nn.Parameter(torch.zeros(basis.shape[1], width, width))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32), persistent=False)
        self.register_buffer("degrees_of", degree_index(degrees * degrees), persistent=False)
        self.register_buffer("partner", torch.tensor(partner_index(degrees)), persistent=False)
        scale = torch.tensor(signed_orders(degrees) / degrees, dtype=torch.float32)
        self.register_buffer("zonal_scale", scale, persistent=False)

    def degree_weights(self):
        isotropic = torch.einsum("ln,ncd->lcd", self.basis, self.isotropic)
        zonal = torch.einsum("ln,ncd->lcd", self.basis, self.zonal)
        return torch.cat([isotropic, zonal], dim=1)[self.degrees_of]

    def mix(self, coefficients):
        swapped = coefficients[..., self.partner, :] * self.zonal_scale.to(coefficients.dtype)[:, None]
        stacked = torch.cat([coefficients, swapped], dim=-1)
        count, width = stacked.shape[-2:]
        flat = stacked.movedim(-2, 0).reshape(count, -1, width)
        mixed = torch.bmm(flat, self.degree_weights().to(flat.dtype))
        return mixed.reshape(count, *stacked.shape[:-2], -1).movedim(0, -2)


def slepian_degree_operators(transform, basis, scale):
    count = transform.coefficients
    isotropic = np.zeros((basis.shape[1], count, count))
    zonal = np.zeros((basis.shape[1], count, count))
    for degree, block in enumerate(transform.degree_blocks()):
        orders = np.arange(-degree, degree + 1) / scale
        gram = block.T @ block
        derivative = block.T @ (orders[:, None] * block[::-1])
        isotropic += basis[degree][:, None, None] * gram[None]
        zonal += basis[degree][:, None, None] * derivative[None]
    return np.concatenate([isotropic, zonal])


class SlepianDegreeChannelMixing(SpectralStage):
    def __init__(self, transform, width, knots, pretrained_degrees):
        super().__init__(transform)
        basis, pretrained = regional_basis(transform.degrees, knots, pretrained_degrees)
        added = basis.shape[1] - pretrained
        operators = slepian_degree_operators(transform, basis, pretrained_degrees)
        self.register_buffer("operators", torch.tensor(operators, dtype=torch.float32), persistent=False)
        self.isotropic = nn.Parameter(torch.zeros(pretrained, width, width), requires_grad=False)
        self.zonal = nn.Parameter(torch.zeros(pretrained, width, width), requires_grad=False)
        self.isotropic_new = nn.Parameter(torch.zeros(added, width, width))
        self.zonal_new = nn.Parameter(torch.zeros(added, width, width))
        size = transform.coefficients
        self.delta = nn.Parameter(torch.zeros(size, size))

    def knot_weights(self):
        return torch.cat([self.isotropic, self.isotropic_new, self.zonal, self.zonal_new])

    def mix(self, coefficients):
        weighted = torch.einsum("...jc,kcd->...kjd", coefficients, self.knot_weights().to(coefficients.dtype))
        spectral = torch.einsum("kij,...kjd->...id", self.operators.to(coefficients.dtype), weighted)
        return spectral + torch.matmul(self.delta.mT, coefficients)


def build_channel_mixing(transform, width, knots, pretrained_degrees):
    if not getattr(transform, "regional", False):
        return DegreeChannelMixing(transform, width, knots)
    if pretrained_degrees is None:
        raise ValueError("regional channel mixing needs the pretrained band limit of each level")
    return SlepianDegreeChannelMixing(transform, width, knots, pretrained_degrees)


def build_mixing(transform, rank=None, filter_width=None, per_coefficient=False):
    regional = getattr(transform, "regional", False)
    if regional and filter_width is not None:
        return SlepianChannelMixing(transform, filter_width)
    if regional:
        return SlepianMixing(transform)
    if filter_width is not None and per_coefficient:
        return CoefficientFilteredMixing(transform, filter_width)
    if filter_width is not None:
        return DegreeFilteredMixing(transform, filter_width)
    if rank is None:
        return SpectralMixing(transform)
    return DiagonalLowRankMixing(transform, rank)
