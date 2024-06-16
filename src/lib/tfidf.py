import joblib
from sklearn.pipeline import Pipeline

from .classifier import Classifier
from .config import ClassifierConfig


class TFIDFLogistic(Classifier):

    def __init__(self, data: ClassifierConfig) -> None:
        super().__init__(data)
        self.model: Pipeline = joblib.load("/data/models/tf-idf-logistic.joblib")

    def run(self, df):
        return self.model.predict(df["title"])
