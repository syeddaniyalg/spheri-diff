import torch.nn.functional as functional
from torch import nn

from spheridiff.models.layers.init import zeroed
from spheridiff.models.layers.norm import RMSNorm

GROUP_EPSILON = 1e-6
MLP_RATIO = 4


class ResBlock(nn.Module):
    def __init__(self, channels, embedding_dim, groups):
        super().__init__()
        self.film = nn.Conv2d(embedding_dim, 2 * channels, kernel_size=1)
        self.input_layers = nn.Sequential(
            nn.GroupNorm(groups, channels, eps=GROUP_EPSILON),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.output_norm = nn.GroupNorm(groups, channels, eps=GROUP_EPSILON)
        self.output_layers = nn.Sequential(
            nn.SiLU(),
            zeroed(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
        )

    def forward(self, values, embedding):
        hidden = self.input_layers(values)
        scale, shift = self.film(embedding[:, :, None, None]).chunk(2, dim=1)
        hidden = self.output_norm(hidden) * (1 + scale) + shift
        return values + self.output_layers(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, dimension, heads, embedding_dim, rotary):
        super().__init__()
        self.heads = heads
        self.dimension = dimension
        head_dimension = dimension // heads
        self.rotary = rotary
        self.film = nn.Linear(embedding_dim, 2 * dimension)
        self.norm = RMSNorm(dimension)
        self.mlp_width = MLP_RATIO * dimension
        self.fused = nn.Linear(dimension, 3 * dimension + self.mlp_width)
        self.query_norm = RMSNorm(head_dimension)
        self.key_norm = RMSNorm(head_dimension)
        self.attention_out = zeroed(nn.Linear(dimension, dimension))
        self.mlp_out = nn.Sequential(nn.SiLU(), zeroed(nn.Linear(self.mlp_width, dimension)))

    def attend(self, projected):
        batch, tokens, _ = projected.shape
        shaped = projected.view(batch, tokens, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = shaped.unbind(0)
        query = self.rotary(self.query_norm(query))
        key = self.rotary(self.key_norm(key))
        attended = functional.scaled_dot_product_attention(query, key, value)
        return attended.transpose(1, 2).reshape(batch, tokens, self.dimension)

    def forward(self, tokens, embedding):
        scale, shift = self.film(embedding).chunk(2, dim=-1)
        normed = self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        projected, mlp_hidden = self.fused(normed).split((3 * self.dimension, self.mlp_width), dim=-1)
        tokens = tokens + self.attention_out(self.attend(projected))
        return tokens + self.mlp_out(mlp_hidden)
