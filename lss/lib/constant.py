import numpy as np

from .classifier import Classifier
from .config import ClassifierConfig


class Constant(Classifier):

    def __init__(self, data: ClassifierConfig, isPositive: bool) -> None:
        super().__init__(data)
        self.constant = 1 if isPositive else 0

    def run(self, df):
        return np.array([self.constant] * df.shape[0])
