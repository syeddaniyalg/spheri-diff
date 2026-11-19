class PeriodicEvaluation:
    headline = ("val_loss/mean", "nrmse/mean", "crps_normalised/mean")

    def __init__(self, validation, forecast):
        self.validation = validation
        self.forecast = forecast

    def evaluate(self, model):
        return {**self.validation.evaluate(model), **self.forecast.evaluate(model)}

    def baselines(self):
        return self.forecast.baselines()

    def window_record(self):
        return self.forecast.window_record()
