import os
import xml.etree.ElementTree as ET
from pathlib import Path

import feedparser
import pandas as pd
from feedparser import FeedParserDict
from loguru import logger
from pandas import DataFrame
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..lib.classifier import Classifier
from ..lib.types import Config, FeedConfig
from ..lib.utils import hash_url, load_config, load_models
from ..models import (
    Item,
    Label,
    get_item_by_feedUrl_and_title,
    get_past_two_weeks_items_by_feedUrl,
)

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
    entries: FeedParserDict, feed_config: FeedConfig, session: Session
) -> pd.DataFrame:
    mapped_entries = [
        {key: entry.get(key, default) for key, default in ITEM_PAIRS}
        for entry in entries
    ]
    df = pd.DataFrame(mapped_entries)
    df = df.drop_duplicates(
        subset=["title"]
    )  # Have seen some feeds accidentally double post
    mask = df.apply(
        lambda row: get_item_by_feedUrl_and_title(
            session, feed_config.url, row["title"]
        )
        is None,
        axis=1,
    )
    return df[mask]


def add_predictions(df: DataFrame, models: list[Classifier], quorom: int) -> DataFrame:
    df["votes"] = 0
    for model in models:
        name = model.__class__.__name__
        df[f"prediction_{name}"] = model.run(df)
        df["votes"] += df[f"prediction_{name}"] * model.weight
    df["prediction"] = df["votes"].apply(
        lambda count: Label.POSITIVE if count >= quorom else Label.NEGATIVE
    )
    logger.info(
        "# POSITIVE = {} / # NEGATIVE = {}",
        len(df["votes"] >= quorom),
        len(df["votes"] < quorom),
    )
    return df


def commit_items(df: DataFrame, feed_config: FeedConfig, session: Session) -> None:
    for _, row in df.iterrows():
        try:
            item = Item(
                feedUrl=feed_config.url,
                prediction=row["prediction"],
                label=None,
                title=row.get("title"),
                link=row.get("link"),
                description=row.get("description"),
                author=row.get("author"),
                category=row.get("category"),
                comments=row.get("comments"),
                enclosure=row.get("enclosure"),
                guid=row.get("guid"),
                pubDate=row.get("pubDate"),
            )
            session.add(item)
            session.commit()
        except Exception as e:
            logger.exception(f"Error while adding item {row.get('title')}: {e}")
            session.rollback()


def write_feed(feed, config: Config, feed_config: FeedConfig, session: Session):

    filter = feed_config.filter
    if filter is None:
        filter = not config.cold_start

    rss = ET.Element("rss")
    rss.set("version", "2.0")

    channel = ET.SubElement(rss, "channel")

    for key, default in CHANNEL_PAIRS:
        element = ET.SubElement(channel, key)
        element.text = feed.channel.get(key, default)

    items = get_past_two_weeks_items_by_feedUrl(
        session, feed_config.url, Label.POSITIVE if filter else None
    )
    for item in items:
        item_element = ET.SubElement(channel, "item")
        for key, default in ITEM_PAIRS:
            element = ET.SubElement(item_element, key)
            if key == "title":
                element.text = f"{'&#11088;' if item.prediction is Label.POSITIVE else ' '} {item.title}"
            elif key == "link":
                element.text = f"{config.host}/update/{item.id}/1"
            elif key == "description":
                element.text = f"<a href='{config.host}/update/{item.id}/0'>Click To Dislike</a><br><br>{item.description}"
            else:
                element.text = getattr(item, key, default)

    tree = ET.ElementTree(rss)

    path = Path("/data/feeds") / hash_url(feed_config.url)
    path.mkdir(parents=True, exist_ok=True)
    with open(file=path / "feed.xml", mode="wb") as xml_file:
        logger.info("Writing to {}...", path / "feed.xml")
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)


def main():
    logger.info("Starting generation job...")
    config = load_config()
    models = load_models(config)
    for feed_config in config.feeds:
        try:
            feed = feedparser.parse(url_file_stream_or_string=feed_config.url)
            logger.info("Successfully parsed {}", feed.channel.title)
        except Exception as e:
            logger.exception("Failed parsing {}: {}", feed_config.url, e)
            continue
        if not feed.entries:
            logger.info("No entries for {}", feed.channel.title)
            continue
        with SessionFactory() as session:
            try:
                df = construct_dataframe(feed.entries, feed_config, session)
                if df.empty:
                    logger.info("No new entries for {}", feed.channel.title)
                    continue
                quorom = feed_config.quorom or config.quorom
                df = add_predictions(df, models, quorom)
                logger.info("Example row: {}", df.iloc[0, :])
                commit_items(df, feed_config, session)
                write_feed(feed, config, feed_config, session)
                logger.info("Done generating {}", feed_config.url)
            except Exception as e:
                logger.exception("Error while generating {}: {}", feed.channel.title, e)
                session.rollback()
    logger.info("Generation job completed")
