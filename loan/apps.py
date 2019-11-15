from django.apps import AppConfig


class LoanConfig(AppConfig):
    name = 'loan'

    def ready(self):
        from .model.train_model import prediction_models
        prediction_models()
