from contextlib import contextmanager

import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    def current_decay(self):
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model):
        weight = 1.0 - self.current_decay()
        self.updates += 1
        for name, parameter in model.named_parameters():
            self.shadow[name].lerp_(parameter.detach(), weight)

    @torch.no_grad()
    def copy_to(self, model):
        for name, parameter in model.named_parameters():
            parameter.copy_(self.shadow[name])

    @contextmanager
    def applied(self, model):
        backup = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(backup[name])

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.updates = state["updates"]
        self.shadow = {name: value.clone() for name, value in state["shadow"].items()}
