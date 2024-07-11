from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .config import ClassifierConfig


class Classifier(ABC):

    def __init__(self, data: ClassifierConfig):
        self.weight = data.weight
        self.classifier_active = data.classifier_active

    @abstractmethod
    def run(self, df: pd.DataFrame) -> np.array:
        pass
