import torch.nn.functional as functional
from torch import nn

FACTOR = 2


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.group_size = in_channels * FACTOR**2 // out_channels
        self.conv = nn.Conv2d(in_channels, out_channels // FACTOR**2, kernel_size=3, padding=1)

    def forward(self, values):
        main = functional.pixel_unshuffle(self.conv(values), FACTOR)
        shuffled = functional.pixel_unshuffle(values, FACTOR)
        batch, _, height, width = shuffled.shape
        grouped = shuffled.reshape(batch, self.out_channels, self.group_size, height, width)
        return main + grouped.mean(dim=2)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.repeats = out_channels * FACTOR**2 // in_channels
        self.conv = nn.Conv2d(in_channels, out_channels * FACTOR**2, kernel_size=3, padding=1)

    def forward(self, values):
        main = functional.pixel_shuffle(self.conv(values), FACTOR)
        duplicated = values.repeat_interleave(self.repeats, dim=1)
        return main + functional.pixel_shuffle(duplicated, FACTOR)
