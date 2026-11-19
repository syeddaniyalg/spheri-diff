from .backbone import Wind
from .config import wind_config_from_dict


def build_wind(architecture, context):
    return Wind(wind_config_from_dict(architecture), context)
