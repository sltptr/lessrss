import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, func
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from . import db


class Base(DeclarativeBase):
    pass


class Label(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1


class Item(db.Model):
    __tablename__ = "item"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String, unique=True, index=True)
    link: Mapped[str]
    label: Mapped[Optional[Label]]
    description: Mapped[Optional[str]]
    author: Mapped[Optional[str]]
    category: Mapped[Optional[str]]
    comments: Mapped[Optional[str]]
    enclosure: Mapped[Optional[str]]
    guid: Mapped[Optional[str]]
    pubDate: Mapped[Optional[str]]
    source: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    def serialize(self):
        return {
            c: (
                getattr(self, c)
                if c != "label"
                else getattr(self, c).name if getattr(self, c) else None
            )
            for c in inspect(self).attrs.keys()
        }
