import os

from flask import redirect, send_from_directory
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Item, Label


def register_routes(app, engine):

    @app.route("/files/<path:subpath>/<filename>", methods=["GET"])
    def files(subpath, filename):
        return send_from_directory(
            os.path.join("/", "data", "files", subpath), filename
        )

    @app.route("/update/<int:id>/<int:value>", methods=["GET"])
    def update(id, value):
        with Session(engine) as session:
            try:
                stmt = select(Item).filter_by(id=id)
                item = session.execute(stmt).scalar()
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
