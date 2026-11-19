from dataclasses import dataclass


@dataclass(frozen=True)
class RectifiedSchedule:
    alpha_min: float = 1e-3
    sigma_min: float = 1e-3

    def __call__(self, levels):
        alpha = levels * self.alpha_min + (1.0 - levels)
        sigma = levels + (1.0 - levels) * self.sigma_min
        return alpha, sigma
