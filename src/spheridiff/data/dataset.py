import numpy as np
import torch
from torch.utils.data import Dataset

from .auxiliary import time_auxiliary


class WeatherWindows(Dataset):
    def __init__(self, frames, seconds, longitudes, static_maps, frame_count):
        self.frames = frames
        self.seconds = seconds
        self.longitudes = longitudes
        self.static_maps = torch.from_numpy(static_maps)
        self.frame_count = frame_count

    def __len__(self):
        return self.frames.shape[0] - self.frame_count + 1

    def __getitem__(self, index):
        stop = index + self.frame_count
        seconds = self.seconds[index:stop]
        frames = torch.from_numpy(np.ascontiguousarray(self.frames[index:stop]))
        height = frames.shape[-2]
        time_maps = torch.from_numpy(time_auxiliary(seconds, self.longitudes, height))
        static_maps = self.static_maps.unsqueeze(0).expand(self.frame_count, -1, -1, -1)
        return {
            "frames": frames,
            "auxiliary": torch.cat([static_maps, time_maps], dim=1),
            "seconds": torch.from_numpy(seconds),
        }
