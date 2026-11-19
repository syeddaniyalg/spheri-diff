from .transforms import undo_precipitation_transform


class PhysicalConverter:
    def __init__(self, statistics, names, precipitation_channels):
        self.statistics = statistics
        self.names = names
        self.precipitation_channels = precipitation_channels

    def __call__(self, frames):
        restored = self.statistics.denormalise(frames)
        return undo_precipitation_transform(restored, self.names, self.precipitation_channels)
