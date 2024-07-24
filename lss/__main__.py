import os

from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base
from .routes import register_routes

app = Flask(__name__)
engine = create_engine(os.environ["SQLALCHEMY_URL"])
Base.metadata.create_all(bind=engine)
register_routes(app, engine)
app.run(host="0.0.0.0")
