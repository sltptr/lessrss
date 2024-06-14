import os

from flask import Flask, redirect, send_from_directory

from . import db
from .models import Item, Label


def register_routes(app: Flask):

    # Quickly check database contents
    @app.route("/data", methods=["GET"])
    def data():
        return [item.serialize() for item in Item.query.all()]

    # Feeds endpoint
    @app.route("/files/<path:subpath>/<filename>", methods=["GET"])
    def files(subpath, filename):
        directory = os.path.join("/", "app", "data", "files", subpath)
        return send_from_directory(directory, filename)

    @app.route("/update/<int:id>/<int:value>", methods=["GET"])
    def link_handler(id, value):
        item: Item = Item.query.get(id)
        item.label = Label(value)
        db.session.commit()
        if item.label is Label.POSITIVE:
            return redirect(item.link)
        return "OK", 200
