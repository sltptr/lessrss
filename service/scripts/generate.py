from datetime import datetime

import feedparser
import pandas as pd
import PyRSS2Gen
from app import create_app, db
from app.models import Item
from config.settings import load_config
from feedparser import FeedParserDict
from lib.gpt import GPT
from lib.tfidf import TFIDFLogistic

app = create_app()


def convert_to_dataframe(entries: list[FeedParserDict]):
    data = [entry.get("title") for entry in entries]
    return pd.DataFrame(data, columns=["title"])


with app.app_context():
    config = load_config()
    tfidf = gpt = None
    try:
        tfidf = TFIDFLogistic(config.models["tfidf"])
    except Exception as e:
        print(e)
    try:
        gpt = GPT(config.models["gpt"])
    except Exception as e:
        print(e)
    models = [tfidf, gpt]
    for feed, feed_config in [
        (feedparser.parse(item.url), item) for item in config.feeds
    ]:
        df = convert_to_dataframe(feed.entries)
        df["votes"] = 0
        for model in models:
            if not model or not model.active:
                continue
            preds = model.run(df)
            df["votes"] += preds * model.vote_weight
        filter = set(df.index[df["votes"] >= config.quorom])
        rssItems = []
        for i, entry in enumerate(feed.entries):
            title = entry.get("title")
            link = entry.get("link")
            description = entry.get("description")
            author = entry.get("author")
            comments = entry.get("comments")
            enclosure = entry.get("enclosure")
            guid = entry.get("guid")
            pubDate = entry.get("guid")
            source = entry.get("source")
            existing_item = Item.query.filter_by(title=entry.title).first()
            if existing_item:
                continue
            item = Item(
                title=title,
                link=link,
                description=description,
                author=author,
                comments=comments,
                enclosure=enclosure,
                guid=guid,
                pubDate=pubDate,
                source=source,
            )
            db.session.add(item)
            db.session.commit()
            if i in filter or feed_config.show_all:
                rssItem = PyRSS2Gen.RSSItem(
                    title=f"{'\u2B50' if i in filter else ''} {title}",
                    link=f"{config.host}/update/{item.id}/1",
                    description=f"{description}<br><br><a href='{config.host}/update/{item.id}/0'>Click To Dislike</a>",
                    author=author,
                    comments=comments,
                    enclosure=enclosure,
                    guid=guid,
                    pubDate=pubDate,
                    source=source,
                )
                rssItems.append(rssItem)

        rss = PyRSS2Gen.RSS2(
            title=f"{feed.feed.title} (RecoRSS)",
            link=feed.feed.link,
            description=feed.feed.description,
            lastBuildDate=datetime.now(),
            items=rssItems,
        )
        with open(
            f"/app/data/files/{feed_config.directory}/feed.xml", "w", encoding="utf-8"
        ) as f:
            rss.write_xml(f)
