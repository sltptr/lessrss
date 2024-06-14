import os

import feedparser
import pandas as pd
from feedgen.feed import FeedGenerator

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


def filter(models, url, host, quorom):
    feed = feedparser.parse(url)
    feed, entries = feed.feed, feed.entries
    fg = FeedGenerator()
    fg.id(feed.get("id"))
    fg.title(feed.get("title"))
    fg.author(feed.get("author"))
    fg.link(href=feed.get("link"), rel="alternate")
    fg.logo(feed.get("logo"))
    fg.subtitle(feed.get("subtitle"))
    fg.language(feed.get("language", "en"))
    df = pd.DataFrame(entries)
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
            items.append(item)
        db.session.add(item)
    db.session.commit()
    for item in items:
        fe = fg.add_entry()
        fe.id(item.id)
        fe.title(
            f"{'\u2B50' if item.prediction is Label.POSITIVE else ''} {item.title}"
        )
        fe.link(href=f"{host}/update/{item.id}/1")
        fe.description(
            f"{item.description}<br><br><a href='{host}/update/{item.id}/0'>Click To Dislike</a>"
        )
        fe.author(name=item.author)
        fe.comments(item.comments)
        fe.enclosure(item.enclosure)
        fe.guid(str(item.guid))
        fe.pubDate(item.pubDate)
        fe.source(item.source)
    return fg.rss_str(pretty=True).decode("utf-8")


def main():
    with app.app_context():
        config = load_config()
        models = load_models(config)
        for feed_config in config.feeds:
            rss = filter(models, feed_config.url, config.host, config.quorom)
            full_path = os.path.join("/app/data/files", feed_config.directory)
            os.makedirs(full_path, exist_ok=True)
            with open(os.path.join(full_path, "feed.xml"), "w", encoding="utf-8") as f:
                f.write(rss)


if __name__ == "__main__":
    main()
