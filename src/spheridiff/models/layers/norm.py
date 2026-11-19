import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dimension, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, values):
        scale = torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return (values.float() * scale).type_as(values) * self.weight
