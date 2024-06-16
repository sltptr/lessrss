import enum
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Label(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1


class Item(Base):
    __tablename__ = "item"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String, unique=True, index=True)
    link: Mapped[str]
    prediction: Mapped[Label]
    label: Mapped[Label | None]
    description: Mapped[str | None]
    author: Mapped[str | None]
    category: Mapped[str | None]
    comments: Mapped[str | None]
    enclosure: Mapped[str | None]
    guid: Mapped[str | None]
    pubDate: Mapped[str | None]
    source: Mapped[str | None]
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
                if c not in ("prediction", "label")
                else getattr(self, c).name if getattr(self, c) else None
            )
            for c in inspect(self).attrs.keys()
        }
