import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(os.environ["SQLALCHEMY_URL"])

Session = sessionmaker(bind=engine)
