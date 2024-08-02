from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .types import ClassifierConfig


class Classifier(ABC):

    def __init__(self, config: ClassifierConfig):
        self.weight = config.weight
        self.active = config.active

    @abstractmethod
    def run(self, df: pd.DataFrame) -> np.array:
        pass
