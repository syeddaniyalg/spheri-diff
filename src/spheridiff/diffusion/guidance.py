import torch

from .covariance import IdentityCovariance
from .solvers import solve_linear


class MMPSDenoiser:
    def __init__(
        self,
        base,
        schedule,
        operator,
        observation,
        noise_variance,
        iterations=2,
        solver="gmres",
        covariance=None,
    ):
        self.base = base
        self.schedule = schedule
        self.operator = operator
        self.observation = observation
        self.noise_variance = noise_variance
        self.iterations = iterations
        self.solver = solver
        self.covariance = covariance or IdentityCovariance()

    def __call__(self, noisy, auxiliary, level):
        alpha, sigma = self.schedule(level)
        gamma = sigma**2 / alpha
        with torch.enable_grad():
            state = noisy.detach().requires_grad_(True)
            estimate = self.base(state, auxiliary, level)
            predicted = self.operator.apply(estimate)

        def spread(vector):
            pulled = torch.autograd.grad(estimate, state, vector, retain_graph=True)[0]
            return gamma * self.covariance.apply(pulled)

        def system(vector):
            adjoint = self.operator.adjoint(vector)
            return self.noise_variance * vector + self.operator.apply(spread(adjoint))

        residual = self.observation - predicted.detach()
        solution = solve_linear(system, residual, self.iterations, self.solver)
        correction = spread(self.operator.adjoint(solution))
        return (estimate + correction).detach()
