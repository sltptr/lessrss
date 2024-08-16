from pydantic import BaseModel, field_validator


class ClassifierConfig(BaseModel):
    weight: int = 1
    active: bool = True


class FeedConfig(BaseModel):
    url: str
    filter: bool | None = None
    quorom: int | None = None


class Config(BaseModel):
    host: str
    db_url: str
    feeds: list[FeedConfig]
    classifiers: dict[str, ClassifierConfig]
    iam_role: str | None = None
    cold_start: bool = False


class ParsedChannel(BaseModel):
    title: str
    link: str
    description: str


class ParsedItem(BaseModel):
    title: str
    link: str
    description: str | None = None
    author: str | None = None
    comments: str | None = None
    enclosure: str | None = None
    guid: str | None = None
    pubDate: str | None = None

    @field_validator("guid", mode="before")
    @classmethod
    def parse_guid(cls, v):
        if isinstance(v, dict):
            return v["#text"]
        return v
