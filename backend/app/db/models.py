from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from ..db.db import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "test"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(Text)
    chunk = Column(Text)
    embedding = Column(Vector(1536))
