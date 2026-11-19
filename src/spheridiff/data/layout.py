import os
from pathlib import Path

import numpy as np


def write_array(path, array):
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


class CacheLayout:
    def __init__(self, root):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.times_dir = self.root / "times"
        self.static_path = self.root / "static.npz"
        self.stats_path = self.root / "stats.npz"

    def frames_path(self, year):
        return self.frames_dir / f"{year}.npy"

    def times_path(self, year):
        return self.times_dir / f"{year}.npy"

    def has_year(self, year):
        return self.frames_path(year).exists() and self.times_path(year).exists()

    def create(self):
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.times_dir.mkdir(parents=True, exist_ok=True)
