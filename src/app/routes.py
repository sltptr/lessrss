import os

from flask import Flask, redirect, send_from_directory
from sqlalchemy import select

from .database import Session
from .models import Item, Label


def register_routes(app: Flask):

    # Quickly check database contents
    @app.route("/data", methods=["GET"])
    def data():
        session = Session()
        statement = select(Item)
        return [item.serialize() for item in session.scalars(statement).all()]

    # Feeds endpoint
    @app.route("/files/<path:subpath>/<filename>", methods=["GET"])
    def files(subpath, filename):
        directory = os.path.join("/", "data", "files", subpath)
        return send_from_directory(directory, filename)

    @app.route("/update/<int:id>/<int:value>", methods=["GET"])
    def link_handler(id, value):
        session = Session()
        try:
            statement = select(Item).filter_by(id=id)
            item: Item = session.scalars(statement).first()
            item.label = Label(value)
            session.commit()
            if item.label is Label.POSITIVE:
                return redirect(item.link)
            return "OK", 200
        except:
            session.rollback()
            raise
