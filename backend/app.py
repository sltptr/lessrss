import enum
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import feedparser
import joblib
import pandas as pd
import yaml
from feedparser import FeedParserDict
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI
from pydantic import BaseModel, RootModel
from sklearn.pipeline import Pipeline
from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

### Config


class ModelData(BaseModel):
    name: str
    vote_weight: int
    active: bool


class ModelDataList(RootModel):
    root: list[ModelData]


class Feed(BaseModel):
    url: str


class FeedList(RootModel):
    root: list[Feed]


class Config(BaseModel):
    feeds: FeedList
    models: ModelDataList


def load_config() -> Config:
    with open("/app/data/config.yml", "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


### Database


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/rss.db"
db.init_app(app)


class Sentiment(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1
    NONE = 2


class Item(db.Model):
    __tablename__ = "item"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    link: Mapped[str]
    sentiment: Mapped[Sentiment] = mapped_column(default=Sentiment.NONE)
    description: Mapped[Optional[str]]
    author: Mapped[Optional[str]]
    category: Mapped[Optional[str]]
    comments: Mapped[Optional[str]]
    enclosure: Mapped[Optional[str]]
    guid: Mapped[Optional[str]]
    pubDate: Mapped[Optional[str]]
    source: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )


with app.app_context():
    db.create_all()


### ML Models


class Model(ABC):

    def __init__(self, data: ModelData) -> None:
        self.vote_weight = data.vote_weight
        self.active = data.active

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def run(self, df: pd.DataFrame):
        pass


class TFIDFLogistic(Model):

    def __init__(self, data: ModelData) -> None:
        super().__init__(data)

    def load(self):
        self.model: Pipeline = joblib.load("/app/data/tf-idf-logistic.joblib")

    def run(self, df):
        return self.model.predict(df)


class GPT(Model):

    def __init__(self, data: ModelData) -> None:
        super().__init__(data)

    def load(self):
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def run(self, df):
        pass


### Filtering


def download_feed(feed: Feed) -> FeedParserDict:
    return feedparser.parse(feed.url)


def download_all() -> list[FeedParserDict]:
    config = load_config()
    return [download_feed(item) for item in config.feeds.root]


def filter_feed(feed: Feed) -> Feed:
    return feed


def filter_all():
    return


### Routes


# Get raw feeds before filtering is applied
@app.route("/raw", methods=["GET"])
def raw():
    return download_all()


# Get filtered feeds
@app.route("/filter", methods=["GET"])
def filter():
    items = data()
    return download_all()


# Get database
@app.route("/data", methods=["GET"])
def data():
    return [
        {
            "id": item.id,
            "created_at": item.created_at,
            "title": item.title,
            "sentiment": item.sentiment.name,
        }
        for item in Item.query.all()
    ]


# Generate personalized feed
@app.route("/generate", methods=["GET"])
def generate():
    return "todo"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
