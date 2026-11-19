import torch

EPSILON = 1e-20


def inner_product(first, second):
    totals = (first * second).flatten(1).sum(dim=1)
    return totals.reshape(-1, *([1] * (first.ndim - 1)))


def conjugate_gradient(system, rhs, iterations):
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = rhs.clone()
    energy = inner_product(residual, residual)
    for _ in range(iterations):
        image = system(direction)
        step = energy / inner_product(direction, image).clamp_min(EPSILON)
        solution = solution + step * direction
        residual = residual - step * image
        new_energy = inner_product(residual, residual)
        direction = residual + new_energy / energy.clamp_min(EPSILON) * direction
        energy = new_energy
    return solution


class MMPSDenoiser:
    def __init__(self, base, schedule, operator, observation, noise_variance, iterations=2):
        self.base = base
        self.schedule = schedule
        self.operator = operator
        self.observation = observation
        self.noise_variance = noise_variance
        self.iterations = iterations

    def __call__(self, noisy, auxiliary, level):
        alpha, sigma = self.schedule(level)
        gamma = sigma**2 / alpha
        with torch.enable_grad():
            state = noisy.detach().requires_grad_(True)
            estimate = self.base(state, auxiliary, level)
            predicted = self.operator.apply(estimate)

        def covariance(vector):
            adjoint = self.operator.adjoint(vector)
            pulled = torch.autograd.grad(estimate, state, adjoint, retain_graph=True)[0]
            return self.noise_variance * vector + self.operator.apply(gamma * pulled)

        residual = self.observation - predicted.detach()
        solution = conjugate_gradient(covariance, residual, self.iterations)
        correction = gamma * torch.autograd.grad(predicted, state, solution)[0]
        return (estimate + correction).detach()
