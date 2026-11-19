import torch

TEMPERATURE_2M = "2m_temperature"
TEMPERATURE_2M_WEIGHT = 1.0
SURFACE_WEIGHT = 0.1


def pressure_level(name):
    prefix = name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def single_channel_weight(name, level_total):
    level = pressure_level(name)
    if level is not None:
        return level / level_total
    return TEMPERATURE_2M_WEIGHT if name == TEMPERATURE_2M else SURFACE_WEIGHT


def channel_weights(names, source_levels):
    level_total = float(sum(source_levels))
    values = [single_channel_weight(name, level_total) for name in names]
    return torch.tensor(values, dtype=torch.float32)


def loss_weights(grid, names, statistics, source_levels):
    channel = channel_weights(names, source_levels)
    difference = torch.as_tensor(statistics.loss_weights(), dtype=torch.float32)
    area = torch.as_tensor(grid.area_weights, dtype=torch.float32)
    return (channel * difference)[:, None, None] * area[None]
