"""
数据库持久层
SQLAlchemy + SQLite (开发) / PostgreSQL (生产)
"""
from database.session import init_db, get_session, AsyncSessionLocal, engine
from database.models import Base, ChatRecord, EvalRecord, KnowledgeDoc
from database.repository import ChatRepository, EvalRepository, DocRepository

__all__ = [
    "init_db",
    "get_session",
    "AsyncSessionLocal",
    "engine",
    "Base",
    "ChatRecord",
    "EvalRecord",
    "KnowledgeDoc",
    "ChatRepository",
    "EvalRepository",
    "DocRepository",
]
