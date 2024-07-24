import os
import xml.etree.ElementTree as ET

import feedparser
import pandas as pd
from feedparser import FeedParserDict
from loguru import logger
from pandas import DataFrame
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from ...lib.config import FeedConfig
from ...lib.utils import load_config, load_models
from ...models import Item, Label

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
SessionFactory = sessionmaker(bind=engine)


"""
FeedParserDict uses special logic to handle mapping keys from Atom->RSS. 
This means pandas cannot infer all the necessary columns, hence why this method exists.
"""


def construct_dataframe(entries: feedparser.FeedParserDict) -> pd.DataFrame:
    mapped_entries = []
    for entry in entries:
        mapped_entries.append(
            {key: entry.get(key, default) for key, default in ITEM_PAIRS}
        )
    return pd.DataFrame(mapped_entries)


def remove_duplicates(df: pd.DataFrame, session: Session) -> pd.DataFrame:
    def title_exists(title):
        return session.execute(select(Item).where(Item.title == title)).first()

    df = df.drop_duplicates(subset=["title"])
    new_titles = [title for title in df["title"] if not title_exists(title)]
    return df[df["title"].isin(new_titles)].copy()


def create_items(
    df: DataFrame, feed_config: FeedConfig, session: Session
) -> list[Item] | None:
    items = []
    for _, row in df.iterrows():
        try:
            item = Item(
                title=row.get("title"),
                link=row.get("link"),
                prediction=row.get("prediction"),
                label=None,
                description=row.get("description"),
                author=row.get("author"),
                category=row.get("category"),
                comments=row.get("comments"),
                enclosure=row.get("enclosure"),
                guid=row.get("guid"),
                pubDate=row.get("pubDate"),
                source=feed_config.url,
            )
            if item.prediction is Label.POSITIVE or feed_config.show_all:
                items.append(item)
            session.add(item)
        except Exception as e:
            logger.error(f"Error while adding item {row.get('title')}: {e}")
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


config = load_config()
models = load_models(config)
for feed_config in config.feeds:
    quorom = feed_config.quorom or config.quorom
    feed: FeedParserDict = feedparser.parse(url_file_stream_or_string=feed_config.url)
    channel, entries = feed.channel, feed.entries
    if not entries:
        logger.info(f"No entries for {feed_config.url}")
        continue
    raw_df = construct_dataframe(entries)
    try:
        with SessionFactory() as session:
            df = remove_duplicates(raw_df, session)
            if df.empty:
                logger.info(f"No new entries for {feed_config.url}")
                continue
            df["votes"] = 0
            for model in models:
                preds = model.run(df)
                df["votes"] += preds * model.weight
            df["prediction"] = df["votes"].apply(
                lambda x: Label.POSITIVE if x >= quorom else Label.NEGATIVE
            )
            items = create_items(df, feed_config, session)
            dir_path = os.path.join("/data/files", feed_config.directory)
            os.makedirs(name=dir_path, exist_ok=True)
            create_feed(channel, items, config.host, dir_path)
    except Exception as e:
        logger.error(f"Error while handling {feed_config.url}: {e}")
