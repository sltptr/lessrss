import os
import xml.etree.ElementTree as ET

import feedparser
import pandas as pd
from feedparser import FeedParserDict
from loguru import logger
from pandas import DataFrame
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lss.lib.classifier import Classifier

from ...lib.config import FeedConfig
from ...lib.utils import load_config, load_models
from ...models import Item, Label, get_item_by_title_and_source

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


def construct_dataframe(
    channel_title: str, entries: feedparser.FeedParserDict, session: Session
) -> pd.DataFrame:
    mapped_entries = []
    for entry in entries:
        mapped_entries.append(
            {key: entry.get(key, default) for key, default in ITEM_PAIRS}
        )
    df = pd.DataFrame(mapped_entries)
    logger.info("{} shape after `construct_dataframe`: {}", channel_title, df.shape)
    df = df.drop_duplicates(
        subset=["title"]
    )  # Have seen some feeds accidentally double post
    logger.info("{} shape after `drop_duplicates`: {}", channel_title, df.shape)
    mask = df.apply(
        lambda row: get_item_by_title_and_source(session, row["title"], row["source"])
        is None,
        axis=1,
    )
    df = df[mask]
    logger.info(
        "{} shape after `get_item_by_title_and_source` mask was applied: {}",
        channel_title,
        df.shape,
    )
    return df


def add_predictions(df: DataFrame, models: list[Classifier], quorom: int):
    df["votes"] = 0
    for model in models:
        name = model.__class__.__name__
        df[f"prediction_{name}"] = model.run(df)
        df["votes"] += df[f"prediction_{name}"] * model.weight
    df["prediction"] = df["votes"].apply(
        lambda count: Label.POSITIVE if count >= quorom else Label.NEGATIVE
    )


def create_items(
    df: DataFrame, feed_config: FeedConfig, session: Session
) -> list[Item] | None:
    items = []
    logger.info("Writing to {}\n{}", feed_config.url, df)
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
            logger.info("{}, {}", item.source, item.title)
            session.add(item)
        except Exception as e:
            logger.exception(f"Error while adding item {row.get('title')}: {e}")
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


logger.info("Starting generation job...")
config = load_config()
models = load_models(config)
for feed_config in config.feeds:
    try:
        feed: FeedParserDict = feedparser.parse(
            url_file_stream_or_string=feed_config.url
        )
        channel, entries = feed.channel, feed.entries
        logger.info("Successfully parsed {}", channel.title)
    except Exception as e:
        logger.exception("Error while parsing {}: {}", feed_config.url, e)
        continue
    if not entries:
        logger.info("No entries for {}", channel.title)
        continue
    with SessionFactory() as session:
        try:
            df = construct_dataframe(channel.title, entries, session)
            if df.empty:
                logger.info("No new entries for {}", channel.title)
                continue
            quorom = feed_config.quorom or config.quorom
            add_predictions(df, models, quorom)
            dir_path = os.path.join("/data/files", feed_config.directory)
            os.makedirs(name=dir_path, exist_ok=True)
            items = create_items(df, feed_config, session)
            create_feed(channel, items, config.host, dir_path)
            session.commit()
        except Exception as e:
            logger.exception("Error while generating {}: {}", channel.title, e)
            session.rollback()
logger.info("Generation job completed")
