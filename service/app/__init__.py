from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from utils.logging import logger

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/rss.db"
    db.init_app(app)
    with app.app_context():
        from .routes import register_routes

        register_routes(app)
        db.create_all()
    logger.info("App Created")
    return app
