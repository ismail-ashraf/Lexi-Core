from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped,mapped_column
from sqlalchemy import DateTime, func,Integer,String,Text,ForeignKey,UniqueConstraint
from datetime import datetime


class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = 'documents'

    id : Mapped[int] = mapped_column(Integer , primary_key=True , index=True)

    filename : Mapped[str] = mapped_column(String(255),nullable=False)

    storage_path : Mapped[str] = mapped_column(Text,nullable=False)

    content_hash : Mapped[str] = mapped_column(String(64),nullable=False,unique=True)

    status : Mapped[str] = mapped_column(String(20),nullable=False , default='pending')

    attempt_count : Mapped[int] = mapped_column(Integer , nullable=False , default=0)

    last_error : Mapped[str | None] = mapped_column(Text , nullable=True)

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True,server_default=func.now())




class Chunk(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunk_document_index",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
