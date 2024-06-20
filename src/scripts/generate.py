import os
from datetime import datetime

import feedparser
import pandas as pd
import PyRSS2Gen as RSS
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Item, Label
from lib.classifier import Classifier
from lib.config import Config, load_config
from lib.constant import Constant
from lib.gpt import GPT
from lib.tfidf import TFIDFLogistic

engine: Engine = create_engine(url=os.environ["SQLALCHEMY_URL"])
Session = sessionmaker(bind=engine)


# FeedParserDict uses special lagic to handle mapping keys
# from Atom->RSS, and in this use case pandas cannot infer
# all the necessary columns, which is why this method exists.
def map_entries_dataframe(entries: feedparser.FeedParserDict) -> pd.DataFrame:
    keys = [
        "title",
        "link",
        "description",
        "author",
        "comments",
        "enclosure",
        "guid",
        "pubDate",
        "source",
    ]
    mapped_entries = []
    for entry in entries:
        mapped_entries.append({key: entry.get(key) for key in keys})
    return pd.DataFrame(mapped_entries)


def construct_rss_feed(
    models: list[Classifier], url: str, host: str, quorom: int, show_all: bool
) -> RSS.RSS2 | None:
    session = Session()
    try:
        feed: feedparser.FeedParserDict = feedparser.parse(
            url_file_stream_or_string=url
        )
        channel, entries = feed["channel"], feed["items"]
        df = map_entries_dataframe(entries)
        df["votes"] = 0
        for model in models:
            preds = model.run(df)
            df["votes"] += preds * model.vote_weight
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
                source=row.get("source", url.split("//")[1]),
                prediction=(
                    Label.POSITIVE if row["votes"] >= quorom else Label.NEGATIVE
                ),
            )
            if item.prediction is Label.POSITIVE or show_all:
                items.append(item)
            session.add(item)
    except Exception as e:
        print(f"Error occurred: {e}")
        session.rollback()
        return None
    else:
        session.commit()
        return RSS.RSS2(
            title=channel["title"],
            link=channel["link"],
            description=channel["description"],
            lastBuildDate=datetime.now(),
            items=[
                RSS.RSSItem(
                    title=f"{'\u2B50' if item.prediction is Label.POSITIVE else ''} {item.title}",
                    link=f"{host}/update/{item.id}/1",
                    description=f"<a href='{host}/update/{item.id}/0'>Click To Dislike</a><br><br>{item.description}",
                    author=item.author,
                    comments=item.comments,
                    enclosure=item.enclosure,
                    guid=item.guid,
                    pubDate=item.pubDate,
                    source=item.source,
                )
                for item in items
            ],
        )
    finally:
        session.close()


def load_models(config: Config) -> list[Classifier]:
    models: list[Classifier] = []
    for classifier, classifier_config in [
        (TFIDFLogistic, config.classifiers["tfidf"]),
        (GPT, config.classifiers["gpt"]),
    ]:
        if not classifier_config.active:
            continue
        try:
            models.append(classifier(classifier_config))
        except Exception as e:
            print(e)
            models.append(Constant(classifier_config, True))
    print(f"Loaded models: {models}")
    return models


def main() -> None:
    config: Config = load_config()
    models: list[Classifier] = load_models(config)
    for feed_config in config.feeds:
        dir_path = os.path.join("/data/files", feed_config.directory)
        os.makedirs(name=dir_path, exist_ok=True)
        rss_object: RSS.RSS2 | None = construct_rss_feed(
            models=models,
            url=feed_config.url,
            host=config.host,
            quorom=config.quorom,
            show_all=feed_config.show_all,
        )
        with open(
            file=os.path.join(dir_path, "feed.xml"), mode="w", encoding="utf-8"
        ) as f:
            rss_object.write_xml(outfile=f, encoding="utf-8")


if __name__ == "__main__":
    main()
