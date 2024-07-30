import enum
from datetime import datetime

from sqlalchemy import DateTime, MetaData, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

constraint_naming_conventions = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
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
    title: Mapped[str] = mapped_column(String, index=True)
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

    __table_args__ = (UniqueConstraint("title", "source"),)


def get_item_by_title_and_source(
    session: Session, title: str, source: str
) -> Item | None:
    return session.query(Item).filter_by(title=title, source=source).one_or_none()
