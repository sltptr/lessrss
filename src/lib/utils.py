import yaml

from lib.classifier import Classifier
from lib.config import Config
from lib.constant import Constant
from lib.distilbert import DistilBERT
from lib.tfidf import TFIDFLogistic


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
            print(e)
            models.append(Constant(classifier_config, True))
    print(f"Loaded models: {models}")
    return models
