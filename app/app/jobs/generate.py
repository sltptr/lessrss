import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import xmltodict
from loguru import logger
from pandas import DataFrame
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..lib.classifier import Classifier
from ..lib.types import Config, FeedConfig, ParsedChannel, ParsedItem
from ..lib.utils import hash_url, load_config, load_models
from ..models import (
    Item,
    Label,
    get_item_by_feed_url_and_title,
    get_past_two_weeks_items_by_feed_url,
)

engine: Engine = create_engine(url=os.environ["SQLALCHEMY_URL"])
SessionFactory = sessionmaker(bind=engine)


def construct_dataframe(
    feed_config: FeedConfig, items: list[ParsedItem], session: Session
) -> pd.DataFrame:
    df = pd.DataFrame([dict(item) for item in items])
    df = df.drop_duplicates(
        subset=["title"]
    )  # Have seen some feeds accidentally double post
    if df.empty:
        return df
    mask = df.apply(
        lambda row: get_item_by_feed_url_and_title(
            session, feed_config.url, row.get("title", "")
        )
        is None,
        axis=1,
    )
    return df[mask].reset_index(drop=True)


def add_predictions(
    df: DataFrame,
    models: list[Classifier],
) -> DataFrame:
    columns = ["poor", "average", "good"]
    scores = pd.DataFrame([[0, 0, 0] for _ in range(df.shape[0])], columns=columns)
    for model in models:
        proba_df = model.run(df)
        for col in columns:
            scores[col] += proba_df[col] * model.weight
    df["predicted_label"] = scores.apply(lambda row: Label(row.argmax()), axis=1)
    return df


def commit_items(df: DataFrame, feed_config: FeedConfig, session: Session) -> None:
    for _, row in df.iterrows():
        try:
            item = Item(
                feed_url=feed_config.url,
                title=row["title"],
                link=row["link"],
                predicted_label=row.get("predicted_label"),
                label=None,
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
            sys.exit(1)


def update_feed(
    config: Config, feed_config: FeedConfig, channel: ParsedChannel, session: Session
):

    filter = feed_config.filter
    if filter is None:
        filter = not config.cold_start

    xml_dict: dict[str, Any] = {"rss": {"@version": "2.0"}}
    xml_dict["rss"]["channel"] = dict(channel)
    xml_dict["rss"]["channel"]["item"] = []
    xml_items = xml_dict["rss"]["channel"]["item"]

    items = get_past_two_weeks_items_by_feed_url(
        session, feed_config.url, [Label.AVERAGE, Label.GOOD] if filter else []
    )

    for item in items:
        parsed_item = ParsedItem(**item.__dict__)
        parsed_item.link = f"{config.host}/update/{item.id}/1"
        emphasize = f"<a href='{config.host}/update/{item.id}/2'>&#11088; Click To Emphasize</a>"
        deemphasize = f"<a href='{config.host}/update/{item.id}/0'>&#128308; Click To De-emphasize</a>"
        parsed_item.description = (
            f"<p>{emphasize} || {deemphasize}</p><br>{item.description}"
        )
        if item.predicted_label is Label.GOOD:
            parsed_item.title = f"&#11088; | {parsed_item.title}"
        elif item.predicted_label is Label.AVERAGE:
            parsed_item.title = f"&#128309; | {parsed_item.title}"
        elif item.predicted_label is Label.POOR:
            parsed_item.title = f"&#128308; | {parsed_item.title}"
        xml_items.append(dict(parsed_item))

    path = Path("/data/feeds") / hash_url(feed_config.url)
    path.mkdir(parents=True, exist_ok=True)
    with open(file=path / "feed.xml", mode="wb") as file:
        logger.info("Writing to {} the following: {}", path / "feed.xml", xml_dict)
        xmltodict.unparse(xml_dict, output=file, pretty=True)


def main():
    logger.info("Starting generation job...")
    config = load_config()
    for feed_config in config.feeds:
        try:
            r = httpx.get(feed_config.url, follow_redirects=True)
            xml_dict = xmltodict.parse(r.text)
            channel = ParsedChannel(**xml_dict["rss"]["channel"])
            items = [
                ParsedItem(**entry) for entry in xml_dict["rss"]["channel"]["item"]
            ]
            logger.info("Successfully parsed {}", channel.title)
        except Exception as e:
            logger.exception("Failed parsing {}: {}", feed_config.url, e)
            continue
        if not items:
            logger.info("No entries for {}", channel.title)
            continue
        with SessionFactory() as session:
            try:
                df = construct_dataframe(
                    feed_config=feed_config, items=items, session=session
                )
                if df.empty:
                    logger.info("No new entries for {}", channel.title)
                    continue
                if not config.cold_start:
                    models = load_models(config)
                    df = add_predictions(df, models)
                logger.info("{} - First Row: {}", channel.title, df.iloc[0, :])
                commit_items(df=df, feed_config=feed_config, session=session)
                update_feed(
                    config=config,
                    feed_config=feed_config,
                    channel=channel,
                    session=session or True,
                )
                logger.info("Done generating {}", feed_config.url)
            except Exception as e:
                logger.exception("Error while generating {}: {}", channel.title, e)
                session.rollback()
    logger.info("Generation job completed")
