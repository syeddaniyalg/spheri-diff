import torch


def trainer_state(trainer):
    return {
        "step": trainer.step,
        "elapsed": trainer.elapsed,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "ema": trainer.ema.state_dict(),
    }


def load_state(path, device):
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def restore_state(trainer, state):
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scaler.load_state_dict(state["scaler"])
    trainer.ema.load_state_dict(state["ema"])
    trainer.step = state["step"]
    trainer.elapsed = state["elapsed"]
