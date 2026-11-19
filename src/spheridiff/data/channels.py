LAND_SEA_MASK = "land_sea_mask"


def pressure_channel_name(variable, level):
    return f"{level}_{variable}"


def channel_names(channels):
    pressure = [
        pressure_channel_name(variable, level)
        for variable in channels.pressure_variables
        for level in channels.levels
    ]
    return pressure + list(channels.surface_variables)
