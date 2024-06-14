import logging
import sys
from logging import StreamHandler

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/rss.db"
    db.init_app(app)
    with app.app_context():
        from .routes import register_routes

        register_routes(app)
        db.create_all()
    handler = StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
    )
    app.logger.addHandler(handler)
    app.logger.info("App Created")
    return app
