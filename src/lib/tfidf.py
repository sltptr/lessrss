import joblib
from config.settings import ModelConfig
from sklearn.pipeline import Pipeline

from .base import Model


class TFIDFLogistic(Model):

    def __init__(self, data: ModelConfig) -> None:
        super().__init__(data)
        self.model: Pipeline = joblib.load("/app/data/ml_assets/tf-idf-logistic.joblib")

    def run(self, df):
        return self.model.predict(df["title"])
