import os
import sys
from abc import ABC, abstractmethod

import feedparser
import joblib
import yaml
from feedparser import FeedParserDict
from flask import Flask
from openai import OpenAI
from pydantic import BaseModel, RootModel

sys.path.append("/app/data/")


def load_config():
    with open("/app/data/config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config


### Types


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


### Models


class Model(ABC):

    def __init__(self, data: ModelData) -> None:
        self.vote_weight = data.vote_weight
        self.active = data.active

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def run(self):
        pass


class TFIDFLogistic(Model):

    def __init__(self, data: ModelData) -> None:
        super().__init__(data)

    def load(self):
        self.model = joblib.load("/app/data/tf-idf-logistic.joblib")

    def run(self):
        pass


class GPT(Model):

    def __init__(self, data: ModelData) -> None:
        super().__init__(data)

    def load(self):
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def run(self):
        pass


### Filtering


def download_feed(feed: Feed) -> FeedParserDict:
    return feedparser.parse(feed.url)


def download_all() -> list[FeedParserDict]:
    config = load_config()
    feedList = FeedList(root=config["feeds"])
    return [download_feed(item) for item in feedList.root]


def filter_feed(feed: Feed) -> Feed:
    return feed


def filter_all():
    return


### Routes

app = Flask(__name__)


# Get raw feeds before filtering is applied
@app.route("/raw", methods=["GET"])
def raw():
    return download_all()


# Get filtered feeds
@app.route("/filter", methods=["GET"])
def filter():
    return download_all()


# Generate personalized feed
@app.route("/generate", methods=["GET"])
def generate():
    return "todo"


# Update the viewing pattern CSV
def update():
    return "todo"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
