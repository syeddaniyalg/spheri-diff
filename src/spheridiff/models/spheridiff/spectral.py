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


def build_mixing(transform, rank=None):
    if getattr(transform, "regional", False):
        return SlepianMixing(transform)
    if rank is None:
        return SpectralMixing(transform)
    return DiagonalLowRankMixing(transform, rank)
