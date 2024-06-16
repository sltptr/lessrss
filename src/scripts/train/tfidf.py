import os
import sqlite3

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_training_data():
    conn = sqlite3.connect("/data/rss.db")
    cur = conn.cursor()
    cur.execute("SELECT title,label FROM item WHERE label IS NOT Null")
    df = pd.DataFrame(cur.fetchall(), columns=["title", "label"])
    df["label"] = df["label"].map({"NEGATIVE": 0, "POSITIVE": 1})
    return df


pipeline = Pipeline(
    [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(class_weight="balanced"))]
)

try:
    df = load_training_data()
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
