from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from config.settings import ModelConfig


class Model(ABC):

    def __init__(self, data: ModelConfig) -> None:
        self.vote_weight = data.vote_weight
        self.active = data.active
        self.prompt = data.prompt

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def run(self, df: pd.DataFrame) -> np.array:
        pass
