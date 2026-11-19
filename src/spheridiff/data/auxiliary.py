import numpy as np

from .channels import LAND_SEA_MASK

SECONDS_PER_DAY = 86400.0
POSITION_TERMS = 3
TIME_TERMS = 4


def auxiliary_channel_count(static_fields):
    return len(static_fields) + POSITION_TERMS + TIME_TERMS


def standardise(field):
    spread = field.std()
    if spread == 0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread


def prepare_field(name, field):
    field = np.nan_to_num(field)
    return field if name == LAND_SEA_MASK else standardise(field)


def static_auxiliary(static, field_names):
    latitudes = np.radians(static["latitudes"])[:, None]
    longitudes = np.radians(static["longitudes"])[None, :]
    shape = (latitudes.shape[0], longitudes.shape[1])
    fields = [prepare_field(name, static[name]) for name in field_names]
    position = [
        np.broadcast_to(np.sin(latitudes), shape),
        np.cos(latitudes) * np.cos(longitudes),
        np.cos(latitudes) * np.sin(longitudes),
    ]
    return np.stack(fields + position).astype(np.float32)


def year_fraction(seconds):
    stamps = seconds.astype("datetime64[s]")
    years = stamps.astype("datetime64[Y]")
    start = years.astype("datetime64[s]")
    end = (years + 1).astype("datetime64[s]")
    return (stamps - start) / (end - start)


def day_fraction(seconds, longitudes):
    offsets = longitudes / 360.0 * SECONDS_PER_DAY
    shifted = seconds[:, None] + offsets[None, :]
    return (shifted % SECONDS_PER_DAY) / SECONDS_PER_DAY


def time_auxiliary(seconds, longitudes, height):
    year_angle = 2.0 * np.pi * year_fraction(seconds)
    day_angle = 2.0 * np.pi * day_fraction(seconds, longitudes)
    frames, width = day_angle.shape
    shape = (frames, 2, height, width)
    year_features = np.stack([np.sin(year_angle), np.cos(year_angle)], axis=1)
    day_features = np.stack([np.sin(day_angle), np.cos(day_angle)], axis=1)
    year_maps = np.broadcast_to(year_features[:, :, None, None], shape)
    day_maps = np.broadcast_to(day_features[:, :, None, :], shape)
    return np.concatenate([year_maps, day_maps], axis=1).astype(np.float32)
