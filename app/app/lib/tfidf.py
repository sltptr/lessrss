import joblib
from sklearn.pipeline import Pipeline

from .classifier import Classifier
from .types import ClassifierConfig


class TFIDFLogistic(Classifier):

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__(config)
        self.model: Pipeline = joblib.load("/data/models/tf-idf-logistic.joblib")

    def run(self, df):
        return self.model.predict(df["title"])
