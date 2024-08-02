import enum
from datetime import datetime
from typing import Sequence

import arrow
from sqlalchemy import DateTime, Index, MetaData, UniqueConstraint, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

constraint_naming_conventions = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=constraint_naming_conventions)


class Label(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1


class Item(Base):
    __tablename__ = "item"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    feedUrl: Mapped[str]
    prediction: Mapped[Label]
    label: Mapped[Label | None]
    title: Mapped[str]
    link: Mapped[str]
    description: Mapped[str | None]
    author: Mapped[str | None]
    category: Mapped[str | None]
    comments: Mapped[str | None]
    enclosure: Mapped[str | None]
    guid: Mapped[str | None]
    pubDate: Mapped[str | None]

    __table_args__ = (
        UniqueConstraint("title", "feedUrl"),
        Index(None, "feedUrl", "title"),
        Index(None, "feedUrl", "created_at", "prediction"),
    )


def get_item_by_feedUrl_and_title(
    session: Session, title: str, feedUrl: str
) -> Item | None:
    stmt = select(Item).where(Item.title == title, Item.feedUrl == feedUrl)
    return session.scalars(stmt).one_or_none()


def get_past_two_weeks_items_by_feedUrl(
    session: Session, feedUrl: str, prediction: Label | None = None
) -> Sequence[Item]:
    conditions = [
        Item.feedUrl == feedUrl,
        Item.created_at >= arrow.utcnow().shift(weeks=-2).datetime,
    ]
    if prediction is not None:
        conditions.append(Item.prediction == prediction)
    stmt = select(Item).where(*conditions).order_by(Item.created_at)
    return session.scalars(stmt).all()
