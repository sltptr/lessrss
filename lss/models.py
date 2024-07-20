import enum

from sqlalchemy import Column, DateTime, Enum, Integer, MetaData, String, Table, func


class Click(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1


metadata_obj = MetaData()
item = Table(
    "item",
    metadata_obj,
    Column("item_id", Integer, primary_key=True),
    Column("title", String),
    Column("link", String),
    Column("prediction", Enum(Click)),
    Column("label", Enum(Click)),
    Column("description", String),
    Column("author", String),
    Column("category", String),
    Column("comments", String),
    Column("enclosure", String),
    Column("guid", String),
    Column("pubDate", String),
    Column("source", String),
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now()),
)
