import torch.nn.functional as functional
from torch import nn

from spheridiff.models.layers.init import zeroed
from spheridiff.models.layers.norm import RMSNorm
from spheridiff.models.layers.rope import RotaryEmbedding

from .spectral import SpectralMixing


class ChannelMlp(nn.Module):
    def __init__(self, in_width, out_width):
        super().__init__()
        self.expand = nn.Linear(in_width, out_width)
        self.activation = nn.GELU()
        self.contract = zeroed(nn.Linear(out_width, out_width))

    def forward(self, values):
        return self.contract(self.activation(self.expand(values)))


class TemporalAttention(nn.Module):
    def __init__(self, width, heads, frames):
        super().__init__()
        self.heads = heads
        self.width = width
        self.projection = nn.Linear(width, 3 * width, bias=False)
        self.output = zeroed(nn.Linear(width, width, bias=False))
        self.rotary = RotaryEmbedding(width // heads, (frames,))

    def forward(self, values):
        batch, frames, points, width = values.shape
        sequences = values.permute(0, 2, 1, 3).reshape(batch * points, frames, width)
        shaped = self.projection(sequences).view(batch * points, frames, 3, self.heads, -1)
        query, key, value = shaped.permute(2, 0, 3, 1, 4).unbind(0)
        attended = functional.scaled_dot_product_attention(
            self.rotary(query), self.rotary(key), value
        )
        merged = attended.transpose(1, 2).reshape(batch * points, frames, width)
        result = self.output(merged).reshape(batch, points, frames, width)
        return result.permute(0, 2, 1, 3)


class SpherIDiffBlock(nn.Module):
    def __init__(self, in_width, out_width, heads, frames, transform):
        super().__init__()
        self.spectral_norm = RMSNorm(in_width)
        self.spectral = SpectralMixing(transform)
        self.mlp_norm = RMSNorm(in_width)
        self.mlp = ChannelMlp(in_width, out_width)
        self.shortcut = (
            nn.Identity() if in_width == out_width else nn.Linear(in_width, out_width, bias=False)
        )
        self.attention_norm = RMSNorm(out_width)
        self.attention = TemporalAttention(out_width, heads, frames)

    def forward(self, values):
        mixed = self.spectral(self.spectral_norm(values))
        values = values + mixed.to(values.dtype)
        values = self.shortcut(values) + self.mlp(self.mlp_norm(values))
        return values + self.attention(self.attention_norm(values))
