from pydantic import BaseModel


class ClassifierConfig(BaseModel):
    weight: int
    active: bool = True


class FeedConfig(BaseModel):
    url: str
    filter: bool | None = None
    quorom: int | None = None


class Config(BaseModel):
    host: str
    quorom: int
    feeds: list[FeedConfig]
    classifiers: dict[str, ClassifierConfig]
    cold_start: bool = False
