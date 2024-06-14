from typing import Optional

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    vote_weight: int
    active: bool
    prompt: Optional[str] = None


class FeedConfig(BaseModel):
    url: str
    show_all: bool
    directory: str


class Config(BaseModel):
    quorom: int
    host: str
    feeds: list[FeedConfig]
    models: dict[str, ModelConfig]


def load_config() -> Config:
    with open("/app/data/config.yml", "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)
