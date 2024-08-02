import os
from pathlib import Path
from urllib.parse import urljoin

from flask import redirect, send_from_directory
from sqlalchemy import select
from sqlalchemy.orm import Session

from .lib.utils import hash_url, load_config
from .models import Item, Label


def register_routes(app, engine):

    @app.route("/feeds", methods=["GET"])
    def feeds():
        config = load_config()
        return {
            feed.url: urljoin(config.host, f"feeds/{hash_url(feed.url)}/feed.xml")
            for feed in config.feeds
        }

    @app.route("/feeds/<path:subpath>/<file>", methods=["GET"])
    def fetch(subpath, file):
        basePath = Path("/")
        return send_from_directory(basePath / "data" / "feeds" / subpath, file)

    @app.route("/update/<int:id>/<int:value>", methods=["GET"])
    def update(id, value):
        with Session(engine) as session:
            try:
                stmt = select(Item).where(Item.id == id)
                item = session.scalars(stmt).one_or_none()
                if not item:
                    return f"Item with ID {id} does not exist", 404
                item.label = Label(value)
                session.commit()
                if item.label is Label.POSITIVE:
                    return redirect(item.link)
                return "OK", 200
            except:
                session.rollback()
                raise
