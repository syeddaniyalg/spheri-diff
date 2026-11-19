import torch
from torch import nn

THETA = 10000.0


def axis_feature_counts(head_dimension, axes):
    half = head_dimension // 2
    base, extra = divmod(half, axes)
    counts = [base] * axes
    if axes == 3 and extra == 1:
        counts[0] += 1
    if axes == 3 and extra == 2:
        counts[1] += 1
        counts[2] += 1
    return [2 * count for count in counts]


def axis_angles(features, size):
    if features == 0:
        return torch.zeros(size, 0)
    frequencies = 1.0 / THETA ** (torch.arange(0, features, 2).float() / features)
    angles = torch.outer(torch.arange(size).float(), frequencies)
    return angles.repeat_interleave(2, dim=-1)


def rotary_angles(head_dimension, sizes):
    counts = axis_feature_counts(head_dimension, len(sizes))
    parts = []
    for axis, (features, size) in enumerate(zip(counts, sizes)):
        shape = [1] * len(sizes) + [features]
        shape[axis] = size
        angles = axis_angles(features, size).reshape(shape)
        parts.append(angles.expand(*sizes, features))
    return torch.cat(parts, dim=-1).reshape(-1, head_dimension)


def rotate_half(values):
    pairs = values.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension, sizes):
        super().__init__()
        if head_dimension % 2:
            raise ValueError("the head dimension must be even for rotary embeddings")
        angles = rotary_angles(head_dimension, sizes)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, values):
        cosine = self.cosine.to(values.dtype)
        sine = self.sine.to(values.dtype)
        return values * cosine + rotate_half(values) * sine
