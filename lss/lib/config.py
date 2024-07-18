import yaml
from pydantic import BaseModel


class ClassifierConfig(BaseModel):
    weight: int
    classifier_active: bool = True


class FeedConfig(BaseModel):
    url: str
    directory: str
    show_all: bool = False
    quorom: int | None = None


class Config(BaseModel):
    host: str
    quorom: int
    feeds: list[FeedConfig]
    classifiers: dict[str, ClassifierConfig]
    filter_active: bool = True


def load_config() -> Config:
    with open(file="/config/config.yml", mode="r") as file:
        config = yaml.safe_load(file)
    return Config(**config)
