import torch

ADAPTATIONS = ("full", "delta_and_pointwise", "delta_only")
RESTRICTED_SUFFIX = "spectral.restricted"
DELTA_SUFFIX = "spectral.delta"


def pretrained_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in state:
        return state["ema"]["shadow"]
    return state["model"]


def dense_spectral_weight(weights, prefix):
    if f"{prefix}.weight" in weights:
        return weights[f"{prefix}.weight"].double()
    diagonal = torch.diag(weights[f"{prefix}.diagonal"].double())
    return diagonal + weights[f"{prefix}.left"].double() @ weights[f"{prefix}.right"].double().mT


def restrict(weight, basis):
    coefficients = weight.shape[0]
    if coefficients > basis.shape[0]:
        raise ValueError(f"the pretrained band limit ({coefficients} coefficients) exceeds the regional one")
    top = basis[:coefficients].double()
    return top.mT @ weight @ top


def restrict_per_channel(weight, kappa, basis):
    coefficients = weight.shape[0]
    top = basis[:coefficients].double()
    degrees = torch.arange(coefficients).float().sqrt().floor().long()
    shared = top.mT @ weight @ top
    filters = kappa.double() if kappa.shape[0] == coefficients else kappa.double()[degrees]
    own = torch.einsum("ki,kc,kj->cij", top, filters, top)
    return shared[None] + own


def restricted_value(name, parameter, weights, module):
    prefix = name[: -len(".restricted")]
    weight = dense_spectral_weight(weights, prefix)
    transform = module.get_submodule(prefix).transform
    if weight.shape[0] > transform.degrees**2:
        raise ValueError(f"the pretrained band limit ({weight.shape[0]} coefficients) exceeds the regional one")
    basis = transform.harmonic_basis(weight.shape[0])
    if parameter.ndim == 3:
        if f"{prefix}.kappa" not in weights:
            raise ValueError(f"the pretrained weights have no degree filters for '{prefix}'")
        return restrict_per_channel(weight, weights[f"{prefix}.kappa"], basis).to(parameter.dtype)
    return restrict(weight, basis).to(parameter.dtype)


def transferred_value(name, parameter, weights, module):
    if name.endswith(DELTA_SUFFIX):
        return torch.zeros_like(parameter)
    if name.endswith(RESTRICTED_SUFFIX):
        return restricted_value(name, parameter, weights, module)
    if name not in weights or weights[name].shape != parameter.shape:
        raise ValueError(f"the pretrained weights have no matching entry for '{name}'")
    return weights[name]


@torch.no_grad()
def transfer_weights(model, weights):
    for name, parameter in model.named_parameters():
        parameter.copy_(transferred_value(name, parameter, weights, model))
    return model


def apply_adaptation(model, adaptation):
    if adaptation not in ADAPTATIONS:
        raise ValueError(f"unknown adaptation '{adaptation}', choose from {ADAPTATIONS}")
    regional = any(name.endswith(DELTA_SUFFIX) for name, _ in model.named_parameters())
    if adaptation != "full" and not regional:
        raise ValueError(f"adaptation '{adaptation}' needs a regional SpherIDiff model")
    if adaptation == "delta_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.endswith(DELTA_SUFFIX))
    return model
