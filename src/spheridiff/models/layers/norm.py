import torch
import torch.nn.functional as functional
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dimension, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def reference(self, values):
        scale = torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return (values.float() * scale).type_as(values) * self.weight

    def fused(self, values):
        weight = self.weight.to(values.dtype)
        return functional.rms_norm(values, (values.shape[-1],), weight, self.epsilon)

    def forward(self, values):
        if hasattr(functional, "rms_norm"):
            return self.fused(values)
        return self.reference(values)
