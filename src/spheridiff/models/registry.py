from .spheridiff.build import build_spheridiff
from .spheridiff.pooled import build_pooled
from .wind.build import build_wind

BUILDERS = {"wind": build_wind, "spheridiff": build_spheridiff, "spheridiff_pooled": build_pooled}


def build_model(name, architecture, context):
    if name not in BUILDERS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(BUILDERS)}")
    return BUILDERS[name](architecture, context)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
