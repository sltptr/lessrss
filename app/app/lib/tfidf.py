import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .classifier import Classifier
from .types import ClassifierConfig


class TFIDFLogistic(Classifier):

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)
        self.model: Pipeline = joblib.load("/data/models/tf-idf-logistic.joblib")

    def run(self, df):
        return pd.DataFrame(
            self.model.predict_proba(df["title"]),
            columns=["poor", "average", "good"],
        )
