from abc import ABC, abstractmethod

from pandas import DataFrame

from .types import ClassifierConfig


class Classifier(ABC):

    def __init__(self, config: ClassifierConfig):
        self.weight = config.weight
        self.active = config.active

    @abstractmethod
    def run(self, df: DataFrame) -> DataFrame:
        pass
