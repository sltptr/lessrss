import numpy as np

from .classifier import Classifier
from .types import ClassifierConfig


class Constant(Classifier):

    def __init__(self, config: ClassifierConfig, isPositive: bool) -> None:
        super().__init__(config)
        self.constant = 1 if isPositive else 0

    def run(self, df):
        return np.array([self.constant] * df.shape[0])
