import torch
import torch.nn.functional as functional
from torch import nn

from spheridiff.models.layers.init import zeroed
from spheridiff.models.layers.norm import RMSNorm
from spheridiff.models.layers.rope import RotaryEmbedding, rotate_half

from .spectral import build_channel_mixing, build_mixing


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

    def rotate(self, values, frames):
        cosine = self.rotary.cosine.to(values.dtype).view(1, frames, 1, 1, -1)
        sine = self.rotary.sine.to(values.dtype).view(1, frames, 1, 1, -1)
        return values * cosine + rotate_half(values) * sine

    def forward(self, values):
        batch, frames, points, width = values.shape
        shaped = self.projection(values).view(batch, frames, points, 3, self.heads, -1)
        query, key, value = shaped.unbind(3)
        query, key = self.rotate(query, frames), self.rotate(key, frames)
        scores = torch.einsum("btnhd,bsnhd->bnhts", query, key) * query.shape[-1] ** -0.5
        attended = torch.einsum("bnhts,bsnhd->btnhd", scores.softmax(dim=-1), value)
        return self.output(attended.reshape(batch, frames, points, width))


class SpaceTimeAttention(TemporalAttention):
    def forward(self, values):
        batch, frames, points, width = values.shape
        shaped = self.projection(values).view(batch, frames, points, 3, self.heads, -1)
        query, key, value = shaped.unbind(3)
        query, key = self.rotate(query, frames), self.rotate(key, frames)
        tokens = [part.reshape(batch, frames * points, self.heads, -1).transpose(1, 2) for part in (query, key, value)]
        attended = functional.scaled_dot_product_attention(*tokens)
        return self.output(attended.transpose(1, 2).reshape(batch, frames, points, width))


class SpherIDiffBlock(nn.Module):
    def __init__(
        self,
        in_width,
        out_width,
        heads,
        frames,
        transform,
        rank=None,
        degree_filters=False,
        coefficient_filters=False,
        channel_knots=None,
        pretrained_degrees=None,
    ):
        super().__init__()
        self.spectral_norm = RMSNorm(in_width)
        filter_width = in_width if degree_filters or coefficient_filters else None
        if channel_knots is None:
            self.spectral = build_mixing(transform, rank, filter_width, coefficient_filters)
        else:
            self.spectral = build_channel_mixing(transform, in_width, channel_knots, pretrained_degrees)
        self.mlp_norm = RMSNorm(in_width)
        self.mlp = ChannelMlp(in_width, out_width)
        self.shortcut = (
            nn.Identity() if in_width == out_width else nn.Linear(in_width, out_width, bias=False)
        )
        self.attention_norm = RMSNorm(out_width)
        self.attention = TemporalAttention(out_width, heads, frames)
        self.local = None

    def forward(self, values):
        mixed = self.spectral(self.spectral_norm(values))
        values = values + mixed.to(values.dtype)
        values = self.shortcut(values) + self.mlp(self.mlp_norm(values))
        values = values + self.attention(self.attention_norm(values))
        if self.local is None:
            return values
        return values + self.local(values).to(values.dtype)
