import os

from flask import Flask
from sqlalchemy import create_engine

from .models import Base
from .routes import register_routes

app = Flask(__name__)
engine = create_engine(os.environ["SQLALCHEMY_URL"])
Base.metadata.create_all(bind=engine)
register_routes(app, engine)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
