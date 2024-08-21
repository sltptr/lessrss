import base64
import hashlib
from collections import Counter

import pandas as pd
import yaml
from loguru import logger
from pandas import DataFrame
from sklearn.utils import resample

from app.models import Label

from .classifier import Classifier
from .constant import Constant
from .distilbert import DistilBERT
from .tfidf import TFIDFLogistic
from .types import Config


def load_config() -> Config:
    with open(file="/config/config.yml", mode="r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


def load_models(config: Config) -> list[Classifier]:
    models: list[Classifier] = []
    init = [
        (TFIDFLogistic, config.classifiers["tfidf"]),
        (DistilBERT, config.classifiers["distilbert"]),
    ]
    for cls, cfg in init:
        if not cfg.active:
            continue
        try:
            models.append(cls(cfg))
        except Exception as e:
            logger.exception("Error while loading model: {}", e)
            models.append(Constant(cfg, True))
    logger.info("Loaded models: {}", models)
    return models


def hash_url(url: str, max_len=8):
    sha256 = hashlib.sha256()
    sha256.update(url.encode("utf-8"))
    base64_hashed = base64.urlsafe_b64encode(sha256.digest()).decode("utf-8")
    truncated_hashed_string = base64_hashed[:max_len]
    return truncated_hashed_string


def upsample_dataframe_by_label(df: DataFrame) -> DataFrame:
    n_samples = (
        max(Counter(df["label"]).values()) * 2
    )  # Doubling to improve odds of at least one use of each entry
    resampled: list[DataFrame] = [
        resample(df[df["label"] == cls], replace=True, n_samples=n_samples)
        for cls in df["label"].unique()
    ]
    return pd.concat(resampled, ignore_index=True)
