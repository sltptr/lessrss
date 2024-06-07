import joblib
from config.settings import ModelConfig

from .base import Model


class DistilBERT(Model):

    def __init__(self, data: ModelConfig) -> None:
        super().__init__(data)

    def load(self):
        self.model = joblib.load("/app/data/models/distilbert.joblib")

    def run(self, df):
        return self.model.predict(df["title"])
