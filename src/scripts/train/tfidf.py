import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Item, Label

engine: Engine = create_engine(url=os.environ["SQLALCHEMY_URL"])
Session = sessionmaker(bind=engine)

try:
    with Session() as session:
        statement = select(Item).where(Item.label != None)
        items = session.scalars(statement).all()
        df = pd.DataFrame(
            [
                {"title": item.title, "label": 1 if item.label is Label.POSITIVE else 0}
                for item in items
            ]
        )
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(class_weight="balanced")),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df["title"], df["label"], test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    directory = "/data/models"
    os.makedirs(directory, exist_ok=True)
    file = os.path.join(directory, "tf-idf-logistic.joblib")
    joblib.dump(pipeline, file)
except Exception as e:
    print(e)
