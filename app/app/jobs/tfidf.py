import os
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import sessionmaker

from ..lib.utils import load_config, upsample_dataframe_by_label
from ..models import Item

config = load_config()

engine: Engine = create_engine(url=config.db_url)
Session = sessionmaker(bind=engine)


def main():
    try:
        with Session() as session:
            stmt = select(Item).where(Item.label != None)
            items = session.scalars(stmt).all()
            df = pd.DataFrame(
                [
                    {
                        "title": item.title,
                        "label": item.label.value,
                    }
                    for item in items
                ]
            )
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(class_weight="balanced")),
            ]
        )
        model = OneVsRestClassifier(pipeline)
        train, test = train_test_split(
            df,
            test_size=0.1,
            stratify=df[["label"]],
        )
        upsampled_train = upsample_dataframe_by_label(
            train
        )  # Very important to upsample after split to avoid copying items across test/train
        model.fit(upsampled_train["title"], upsampled_train["label"])
        y_pred = model.predict(test["title"])
        score = f1_score(y_pred, test["label"], average="macro")
        logger.info("F1 Score: {}", score)
        path = Path("/data/models")
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path / "tf-idf-logistic.joblib")
        logger.info("Fitted TFIDF pipeline")
    except Exception as e:
        logger.exception(f"Exception occured: {e}")
