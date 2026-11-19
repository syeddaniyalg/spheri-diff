import torch
from torch import nn


class SpectralMixing(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        size = transform.coefficients
        self.weight = nn.Parameter(torch.zeros(size, size))

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            coefficients = torch.matmul(self.transform.analysis.mT, values.float())
            mixed = torch.matmul(self.weight.mT, coefficients)
            return torch.matmul(self.transform.synthesis.mT, mixed)
