import torch

from .settings import TASKS


class FullEvaluation:
    def __init__(self, forecast, tasks):
        self.forecast = forecast
        self.tasks = tasks

    def run_forecast(self, model):
        record = self.forecast.evaluate(model, include_ssr=True)
        return {f"forecast/{key}": value for key, value in record.items()}

    def run_tasks(self, model, names):
        record = {}
        for task in names:
            record.update(self.tasks.evaluate(model, task))
        return record

    def run(self, model, names=TASKS, include_forecast=True):
        model.eval()
        with torch.no_grad():
            record = self.run_forecast(model) if include_forecast else {}
        record.update(self.run_tasks(model, names))
        return record
