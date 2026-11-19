import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}


def resolve_precision(precision, device):
    return "fp32" if precision == "fp16" and device.type != "cuda" else precision


def autocast_dtype(precision):
    return DTYPES[precision]


def build_scaler(precision, device):
    enabled = precision == "fp16" and device.type == "cuda"
    return torch.amp.GradScaler(device.type, enabled=enabled)


def autocast_context(precision, device):
    dtype = autocast_dtype(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)
