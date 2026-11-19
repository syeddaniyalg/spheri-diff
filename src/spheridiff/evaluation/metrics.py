import torch


def area_mean(values, area):
    return (values * area).mean(dim=(-2, -1))


def mean_square_error(prediction, truth, area):
    return area_mean((prediction - truth) ** 2, area)


def crps_unbiased(ensemble, truth, area):
    members = ensemble.shape[1]
    skill = (ensemble - truth[:, None]).abs().mean(dim=1)
    ordered, _ = torch.sort(ensemble, dim=1)
    ranks = torch.arange(members, dtype=ensemble.dtype, device=ensemble.device)
    factors = (2.0 * ranks + 1.0 - members).reshape(1, members, 1, 1, 1, 1)
    spread = (ordered * factors).sum(dim=1) / (members * (members - 1))
    return area_mean(skill - spread, area)


def ensemble_variance(ensemble, area):
    return area_mean(ensemble.var(dim=1, unbiased=True), area)


class RunningMean:
    def __init__(self):
        self.total = None
        self.count = 0

    def add(self, values):
        batch_total = values.sum(dim=0)
        self.total = batch_total if self.total is None else self.total + batch_total
        self.count += values.shape[0]

    def mean(self):
        return self.total / self.count
