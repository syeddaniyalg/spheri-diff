from .config import spheridiff_config_from_dict
from .model import SpherIDiff


def build_spheridiff(architecture, context):
    return SpherIDiff(spheridiff_config_from_dict(architecture), context)
