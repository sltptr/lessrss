import os
from datetime import datetime

import feedparser
import pandas as pd
import PyRSS2Gen as RSS

from app import create_app, db
from app.models import Item, Label
from config.settings import load_config
from lib.gpt import GPT
from lib.tfidf import TFIDFLogistic

app = create_app()


def load_models(config):
    models = []
    for model_cls, model_config in [
        (TFIDFLogistic, config.models["tfidf"]),
        (GPT, config.models["gpt"]),
    ]:
        try:
            if model_config.active:
                models.append(model_cls(model_config))
        except Exception as e:
            print(e)
    return models


# FeedParserDict uses special lagic to handle mapping keys
# from Atom->RSS, and in this use case pandas cannot infer
# all the necessary columns, which is why this method exists.
def map_entries_dataframe(entries: feedparser.FeedParserDict):
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


def filter(models, url, host, quorom):
    feed = feedparser.parse(url)
    channel, entries = feed["channel"], feed["items"]
    df = map_entries_dataframe(entries)
    df["votes"] = 0
    for model in models:
        preds = model.run(df)
        df["votes"] += preds * model.vote_weight
    items = []
    for _, row in df.iterrows():
        title = row.get("title")
        link = row.get("link")
        description = row.get("description")
        author = row.get("author")
        comments = row.get("comments")
        enclosure = row.get("enclosure")
        guid = row.get("guid")
        pubDate = row.get("pubDate")
        source = row.get("source", url.split("//")[1])
        votes = row["votes"]
        existing_item = Item.query.filter_by(title=title).first()
        if existing_item:
            continue
        item = Item(
            title=title,
            link=link,
            prediction=(Label.POSITIVE if votes >= quorom else Label.NEGATIVE),
            description=description,
            author=author,
            comments=comments,
            enclosure=enclosure,
            guid=guid,
            pubDate=pubDate,
            source=source,
        )
        if item.prediction is Label.POSITIVE:
            items.append(
                RSS.RSSItem(
                    title=f"{'\u2B50' if votes >= quorom else ''} {title}",
                    link=f"{host}/update/{item.id}/1",
                    description=f"{description}<br><br><a href='{host}/update/{item.id}/0'>Click To Dislike</a>",
                    author=author,
                    comments=comments,
                    enclosure=enclosure,
                    guid=guid,
                    pubDate=pubDate,
                    source=source,
                )
            )
        db.session.add(item)
    db.session.commit()
    return RSS.RSS2(
        title=channel["title"],
        link=channel["link"],
        description=channel["description"],
        lastBuildDate=datetime.now(),
        items=items,
    )


def main():
    with app.app_context():
        config = load_config()
        models = load_models(config)
        for feed_config in config.feeds:
            rss2 = filter(models, feed_config.url, config.host, config.quorom)
            full_path = os.path.join("/app/data/files", feed_config.directory)
            os.makedirs(full_path, exist_ok=True)
            with open(os.path.join(full_path, "feed.xml"), "w", encoding="utf-8") as f:
                rss2.write_xml(f)


if __name__ == "__main__":
    main()
