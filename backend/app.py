import enum
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import feedparser
import joblib
import numpy as np
import pandas as pd
import PyRSS2Gen
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from feedparser import FeedParserDict
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sqlalchemy import DateTime, func
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

### Config


class ModelConfig(BaseModel):
    vote_weight: int
    active: bool
    prompt: Optional[str] = None


class FeedConfig(BaseModel):
    url: str
    directory: str


class Config(BaseModel):
    quorom: int
    feeds: list[FeedConfig]
    models: dict[str, ModelConfig]


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


class Label(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1
    NONE = 2


class Item(db.Model):
    __tablename__ = "item"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    link: Mapped[str]
    label: Mapped[Label] = mapped_column(default=Label.NONE)
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

    def serialize(self):
        return {
            c: getattr(self, c) if c != "label" else getattr(self, c).name
            for c in inspect(self).attrs.keys()
        }


with app.app_context():
    db.create_all()


### ML Models


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


class TFIDFLogistic(Model):

    def __init__(self, data: ModelConfig) -> None:
        super().__init__(data)

    def load(self):
        self.model: Pipeline = joblib.load("/app/data/models/tf-idf-logistic.joblib")

    def run(self, df):
        return self.model.predict(df["title"])


class GPT(Model):

    def __init__(self, data: ModelConfig) -> None:
        super().__init__(data)

    def load(self):
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def run(self, df):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who labels RSS feed items. \
            The user will first give a description of their preferences for RSS items. \
            The user will then give a list of titles for feed items, i.e. ['Post A', 'Post B', 'Post C']. \
            You will then respond with a JSON array of 0s and 1s where 0 means you don't think the user \
            would like the RSS item based on the title, and a 1 means you do think the user would \
            like the item, i.e. [0,1,1] means you think the user wouldn't like Post A but you think \
            that the user would likes Post B and Post C. The JSON response should be in the format \
            {labels: []}.",
                },
                {
                    "role": "user",
                    "content": f"The following is the description of my preferences for RSS items: {self.prompt}. \
                    Here is the list of {df.shape[0]} titles that I need you to process: {df.title.tolist()}.",
                },
            ],
        )
        content_json = json.loads(completion.choices[0].message.content)
        return np.array(content_json["labels"])


def load_tfidf():
    config = load_config()
    return TFIDFLogistic(config.models["tfidf"])


### Routes


def download_all() -> list[tuple[FeedParserDict, str]]:
    config = load_config()
    return [(feedparser.parse(item.url), item.directory) for item in config.feeds]


def convert_to_dataframe(entries: list[FeedParserDict]):
    data = [entry.get("title") for entry in entries]
    return pd.DataFrame(data, columns=["title"])


# Generate personalized feed
@app.route("/generate", methods=["GET"])
def generate():
    config = load_config()
    tfidf = TFIDFLogistic(config.models["tfidf"])
    gpt = GPT(config.models["gpt"])
    models = [tfidf, gpt]
    for feed, directory in download_all():
        df = convert_to_dataframe(feed.entries)
        df["votes"] = 0
        for model in models:
            if not model.active:
                continue
            model.load()
            preds = model.run(df)
            df["votes"] += preds * model.vote_weight
        filter = df.index[df["votes"] >= config.quorom]
        filtered_entries = [feed.entries[i] for i in filter]
        items = []
        for entry in filtered_entries:
            item = PyRSS2Gen.RSSItem(
                title=f"\u2B50 {entry.get("title")}",
                link=entry.get("link"),
                description=entry.get("description"),
                author=entry.get("author"),
                comments=entry.get("comments"),
                enclosure=entry.get("enclosure"),
                guid=entry.get("guid"),
                pubDate=entry.get("pubDate"),
                source=entry.get("source"),
            )
            items.append(item)

        rss = PyRSS2Gen.RSS2(
            title=f"{feed.feed.title} (RecoRSS)",
            link=feed.feed.link,
            description=feed.feed.description,
            lastBuildDate=datetime.now(),
            items=items,
        )
        with open(f"/app/data/files/{directory}/feed.xml", "w", encoding="utf-8") as f:
            rss.write_xml(f)

    return "OK", 200


# Get raw feed before filtering is applied
@app.route("/raw", methods=["GET"])
def raw():
    return download_all()


# Get database
@app.route("/data", methods=["GET"])
def data():
    return [item.serialize() for item in Item.query.all()]


# Feed endpoint
@app.route("/files/<path:subpath>/<filename>", methods=["GET"])
def root(subpath, filename):
    directory = os.path.join("/", "app", "data", "files", subpath)
    return send_from_directory(directory, filename)


### Scheduled Task


scheduler = BackgroundScheduler()
scheduler.add_job(generate, "interval", hours=4)
scheduler.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
