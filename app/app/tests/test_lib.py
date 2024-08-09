import numpy as np
from pandas import DataFrame

from ..lib.distilbert import DistilBERT
from ..lib.tfidf import TFIDFLogistic
from ..lib.types import ClassifierConfig

cfg = ClassifierConfig(weight=1)


def test_tfidf_run():
    df = DataFrame(["one", "two", "three"], columns=["title"])
    model = TFIDFLogistic(cfg)
    df = model.run(df)
    for col in ("proba_poor", "proba_average", "proba_good"):
        assert col in df.columns
        assert np.issubdtype(df[col].dtype, np.float64)
        assert ((df[col] > 0) & (df[col] < 1) & (df[col] != 0.5)).all()


def test_distilbert_run():
    df = DataFrame(["one", "two", "three"], columns=["title"])
    model = DistilBERT(cfg)
    df = model.run(df)
    for col in ("proba_poor", "proba_average", "proba_good"):
        assert col in df.columns
        assert np.issubdtype(df[col].dtype, np.float64)
        assert ((df[col] > 0) & (df[col] < 1) & (df[col] != 0.5)).all()
