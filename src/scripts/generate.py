import os
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import feedparser
import pandas as pd
from feedparser import FeedParserDict
from pandas import DataFrame
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Item, Label
from lib.classifier import Classifier
from lib.config import Config, FeedConfig
from lib.utils import load_config, load_models

CHANNEL_PAIRS = [
    ("title", "No title"),
    ("link", ""),
    ("description", ""),
]

ITEM_PAIRS = [
    ("title", "No title"),
    ("link", ""),
    ("description", ""),
    ("author", ""),
    ("comments", ""),
    ("enclosure", ""),
    ("guid", ""),
    ("pubDate", ""),
    ("source", ""),
]

engine: Engine = create_engine(url=os.environ["SQLALCHEMY_URL"])
Session = sessionmaker(bind=engine)


# FeedParserDict uses special lagic to handle mapping keys
# from Atom->RSS, and in this use case pandas cannot infer
# all the necessary columns, which is why this method exists.
def map_entries_dataframe(entries: feedparser.FeedParserDict) -> pd.DataFrame:
    mapped_entries = []
    for entry in entries:
        mapped_entries.append(
            {key: entry.get(key, default) for key, default in ITEM_PAIRS}
        )
    return pd.DataFrame(mapped_entries)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_items(df: DataFrame, feed_config: FeedConfig, session) -> list[Item] | None:
    items = []
    for _, row in df.iterrows():
        statement = select(Item).filter_by(title=row.get("title"))
        if session.scalars(statement).all():
            continue
        item = Item(
            title=row.get("title"),
            link=row.get("link"),
            description=row.get("description"),
            author=row.get("author"),
            comments=row.get("comments"),
            enclosure=row.get("enclosure"),
            guid=row.get("guid"),
            pubDate=row.get("pubDate"),
            source=row.get("source", feed_config.url.split("//")[1]),
            prediction=(Label.POSITIVE if row["prediction"] else Label.NEGATIVE),
        )
        if item.prediction is Label.POSITIVE or feed_config.show_all:
            items.append(item)
        session.add(item)
    session.commit()
    return items


def create_feed(
    channel_dict: FeedParserDict, items: list[Item], host: str, dir_path: str
):
    rss = ET.Element("rss")
    rss.set("version", "2.0")

    channel = ET.SubElement(rss, "channel")

    for key, default in CHANNEL_PAIRS:
        element = ET.SubElement(channel, key)
        element.text = channel_dict.get(key, default)
    for item in items:
        item_element = ET.SubElement(channel, "item")
        for key, default in ITEM_PAIRS:
            element = ET.SubElement(item_element, key)
            if key == "title":
                element.text = f"{'&#11088;' if item.prediction is Label.POSITIVE else ' '} {item.title}"
            elif key == "link":
                element.text = (
                    f"{host}/update/{item.id}/1"  # Maybe change host to an envvar
                )
            elif key == "description":
                element.text = f"<a href='{host}/update/{item.id}/0'>Click To Dislike</a><br><br>{item.description}"
            else:
                element.text = getattr(item, key, default)

    tree = ET.ElementTree(rss)

    with open(file=os.path.join(dir_path, "feed.xml"), mode="wb") as xml_file:
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)


def main() -> None:
    config: Config = load_config()
    models: list[Classifier] = load_models(config)
    for feed_config in config.feeds:
        feed: FeedParserDict = feedparser.parse(
            url_file_stream_or_string=feed_config.url
        )
        channel, entries = feed["channel"], feed["entries"]
        if not entries:
            print(f"No entries for {feed_config.url}")
            continue
        df = map_entries_dataframe(entries)
        df["votes"] = 0
        for model in models:
            preds = model.run(df)
            df["votes"] += preds * model.vote_weight
        df["prediction"] = df["votes"] >= feed_config.quorom
        print(
            f"Predictions for {feed_config.directory}:\n{df[["title", "votes", "prediction"]]}"
        )
        try:
            with session_scope() as session:
                items = create_items(df, feed_config, session)
                if not items:
                    print(f"Missing items for feed: {feed_config.directory}")
                    continue
                dir_path = os.path.join("/data/files", feed_config.directory)
                os.makedirs(name=dir_path, exist_ok=True)
                create_feed(channel, items, config.host, dir_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
