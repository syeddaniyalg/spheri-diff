from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class GridSpec:
    latitudes: np.ndarray
    longitudes: np.ndarray

    @property
    def height(self):
        return len(self.latitudes)

    @property
    def width(self):
        return len(self.longitudes)

    @property
    def bandlimit(self):
        return min((self.height + 1) // 2, self.width // 2 + 1)

    @property
    def coefficients(self):
        return self.bandlimit**2

    @property
    def latitude_areas(self):
        latitudes = np.radians(self.latitudes)
        spacing = np.radians(np.diff(self.latitudes).mean())
        areas = 2.0 * np.cos(latitudes) * np.sin(spacing / 2.0)
        at_poles = np.isclose(np.abs(latitudes), np.pi / 2.0)
        areas[at_poles] = 2.0 * np.sin(spacing / 4.0) ** 2
        return areas

    @property
    def area_weights(self):
        weights = self.latitude_areas / self.latitude_areas.mean()
        return np.repeat(weights[:, None], self.width, axis=1).astype(np.float32)
