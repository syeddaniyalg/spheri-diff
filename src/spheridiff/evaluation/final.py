import torch


def fingerprint(dataset, indices):
    starts = [int(dataset.seconds[int(index)]) for index in indices]
    return {"indices": [int(index) for index in indices], "start_seconds": starts}


class FinalEvaluation:
    def __init__(self, protocol, forecast, reconstruction, forecast_indices, task_indices):
        self.protocol = protocol
        self.forecast = forecast
        self.reconstruction = reconstruction
        self.windows = {
            "forecast": fingerprint(forecast.dataset, forecast_indices),
            "tasks": fingerprint(reconstruction.dataset, task_indices),
        }

    def forecast_record(self, model):
        record = self.forecast.evaluate(model, include_ssr=True)
        record["window_nmse"] = self.forecast.window_record()["nmse"]
        return record

    def run(self, model):
        model.eval()
        with torch.no_grad():
            forecast = self.forecast_record(model)
            tasks = self.reconstruction.evaluate(model)
        return {
            "protocol": self.protocol.summary(),
            "windows": self.windows,
            "forecast": forecast,
            "tasks": tasks,
        }
