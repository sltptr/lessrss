from datetime import datetime
from enum import Enum
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


class Label(Enum):
    POOR = 0  # Not even interested in the title
    AVERAGE = 1  # Mild interest in the title
    GOOD = 2  # Interested enough to follow the link


class Item(Base):
    __tablename__ = "item"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    feed_url: Mapped[str]
    title: Mapped[str]
    link: Mapped[str]
    predicted_label: Mapped[Label | None]
    label: Mapped[Label | None]
    description: Mapped[str | None]
    author: Mapped[str | None]
    category: Mapped[str | None]
    comments: Mapped[str | None]
    enclosure: Mapped[str | None]
    guid: Mapped[str | None]
    pubDate: Mapped[str | None]

    __table_args__ = (
        UniqueConstraint("title", "feed_url"),
        Index(None, "feed_url", "title"),
        Index(None, "feed_url", "created_at", "predicted_label"),
    )


def get_item_by_feed_url_and_title(
    session: Session, feed_url: str, title: str
) -> Item | None:
    stmt = select(Item).where(Item.title == title, Item.feed_url == feed_url)
    return session.scalars(stmt).one_or_none()


def get_past_two_weeks_items_by_feed_url(
    session: Session, feed_url: str, predicted_labels: list[Label] = []
) -> Sequence[Item]:
    conditions = [
        Item.feed_url == feed_url,
        Item.created_at >= arrow.utcnow().shift(weeks=-2).datetime,
    ]
    if predicted_labels:
        conditions.append(Item.predicted_label.in_(predicted_labels))
    stmt = select(Item).where(*conditions).order_by(Item.created_at)
    return session.scalars(stmt).all()
