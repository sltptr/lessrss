from flask import Flask
from sqlalchemy import create_engine

from app.lib.utils import load_config

from .models import Base
from .routes import register_routes

app = Flask(__name__)
config = load_config()
engine = create_engine(config.db_url)
Base.metadata.create_all(bind=engine)
register_routes(app, engine)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
