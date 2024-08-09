import joblib
from loguru import logger
from sklearn.pipeline import Pipeline

from .classifier import Classifier
from .types import ClassifierConfig


class TFIDFLogistic(Classifier):

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)
        self.model: Pipeline = joblib.load("/data/models/tf-idf-logistic.joblib")

    def run(self, df):
        probabilities = self.model.predict_proba(df["title"])
        df["proba_poor"] = [p[0] for p in probabilities]
        df["proba_average"] = [p[1] for p in probabilities]
        df["proba_good"] = [p[2] for p in probabilities]
        logger.debug(df.iloc[0])
        return df
