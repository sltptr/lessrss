from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .config import ClassifierConfig


class Classifier(ABC):

    def __init__(self, data: ClassifierConfig) -> None:
        self.vote_weight = data.vote_weight
        self.active = data.active
        self.prompt = data.prompt

    @abstractmethod
    def run(self, df: pd.DataFrame) -> np.array:
        pass
