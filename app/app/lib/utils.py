import yaml
from loguru import logger

from .classifier import Classifier
from .config import Config
from .constant import Constant
from .distilbert import DistilBERT
from .tfidf import TFIDFLogistic


def load_config() -> Config:
    with open(file="/config/config.yml", mode="r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


def load_models(config: Config) -> list[Classifier]:
    models: list[Classifier] = []
    classifier_definitions = [
        (TFIDFLogistic, config.classifiers["tfidf"]),
        (DistilBERT, config.classifiers["distilbert"]),
    ]
    for classifier_class, classifier_config in classifier_definitions:
        if not classifier_config.classifier_active:
            continue
        try:
            models.append(classifier_class(classifier_config))
        except Exception as e:
            logger.exception("Error while loading model: {}", e)
            models.append(Constant(classifier_config, True))
    logger.info("Loaded models: {}", models)
    return models
