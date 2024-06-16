from flask import Flask

from .database import Session, engine
from .models import Base
from .routes import register_routes


def create_app():
    app = Flask(__name__)
    register_routes(app)
    Base.metadata.create_all(bind=engine)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0")
