import yaml
from pydantic import BaseModel


class ClassifierConfig(BaseModel):
    vote_weight: int
    active: bool
    prompt: str | None = None


class FeedConfig(BaseModel):
    url: str
    show_all: bool
    directory: str


class Config(BaseModel):
    quorom: int
    host: str
    feeds: list[FeedConfig]
    classifiers: dict[str, ClassifierConfig]


def load_config() -> Config:
    with open(file="/config/config.yml", mode="r") as file:
        config = yaml.safe_load(file)
    return Config(**config)
