"""
SQLAlchemy ORM 模型
定义数据库表结构
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, JSON, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase
import enum


class Base(DeclarativeBase):
    """ORM 基类"""
    pass


def _uuid() -> str:
    """生成字符串型 UUID 主键（String(36)，兼容 SQLite 与 PostgreSQL）"""
    return str(uuid.uuid4())


class QueryType(str, enum.Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    NUMERICAL = "numerical"
    NEGATION = "negation"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class ChatRecord(Base):
    """
    对话历史记录表
    存储每次问答的完整链路数据
    """
    __tablename__ = "chat_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    # 多租户关联（nullable：兼容存量数据与匿名会话）
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    intent = Column(String(32), default="unknown")
    slot_params = Column(JSON, default=dict)
    documents_used = Column(JSON, default=list)      # [doc_id, ...]
    reflection_rounds = Column(Integer, default=0)
    reflection_notes = Column(JSON, default=list)
    conflict_warning = Column(Boolean, default=False)
    conflict_markers = Column(JSON, default=list)

    # 耗时明细
    intent_ms = Column(Integer, default=0)
    retrieve_ms = Column(Integer, default=0)
    generate_ms = Column(Integer, default=0)
    reflection_ms = Column(Integer, default=0)
    total_ms = Column(Integer, default=0)

    # 用量统计（token 消耗 — 用于 /usage/stats 按用户聚合）
    total_tokens = Column(Integer, default=0)

    # 评估指标
    faithfulness = Column(Float, default=0.0)
    answer_relevance = Column(Float, default=0.0)
    context_precision = Column(Float, default=0.0)
    recall_at_5 = Column(Float, default=0.0)

    # 元数据
    trace = Column(JSON, default=list)               # FSM 状态转移追踪
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<ChatRecord(session_id={self.session_id}, intent={self.intent})>"


class EvalRecord(Base):
    """
    评估结果表
    存储批量基准测试的结果
    """
    __tablename__ = "eval_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), nullable=False, index=True)
    session_id = Column(String(64), nullable=False)
    query = Column(Text, nullable=False)
    expected_answer = Column(Text, default="")
    generated_answer = Column(Text, default="")
    query_type = Column(String(32), default="single_hop")

    # 评估分数
    faithfulness = Column(Float, default=0.0)
    answer_relevance = Column(Float, default=0.0)
    context_precision = Column(Float, default=0.0)
    recall_at_5 = Column(Float, default=0.0)

    retrieved_doc_ids = Column(JSON, default=list)
    relevant_doc_ids = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<EvalRecord(session_id={self.session_id}, query_type={self.query_type})>"


class KnowledgeDoc(Base):
    """
    知识库文档元数据表
    管理文档的索引状态和版本
    """
    __tablename__ = "knowledge_docs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=False)
    category = Column(String(64), default="general")
    content_hash = Column(String(64), default="")     # SHA256 用于增量更新检测
    content_length = Column(Integer, default=0)
    indexed_at = Column(DateTime, nullable=True)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<KnowledgeDoc(doc_id={self.doc_id}, title={self.title})>"


# ============================================================
# 多租户 SaaS 表 — 用户 / 知识库 / 文档
# ============================================================

class User(Base):
    """
    用户表
    支持两种注册方式：邮箱密码（hashed_password）+ GitHub OAuth（github_login）
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), default="")   # OAuth 用户为空
    display_name = Column(String(128), default="")
    avatar_url = Column(String(512), default="")
    github_login = Column(String(128), nullable=True, index=True)  # OAuth 身份匹配
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class KnowledgeBase(Base):
    """
    知识库表
    每个知识库拥有独立的 FAISS + BM25 索引（物理隔离，见 processing/indexer.py）
    """
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, name={self.name})>"


class Document(Base):
    """
    文档表
    记录上传文件的解析/索引状态；一个文档切成多个 chunk 写入所属 KB 的索引
    """
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_uuid)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(16), default="")          # pdf / docx / md / txt
    chunk_count = Column(Integer, default=0)
    file_size_bytes = Column(Integer, default=0)
    status = Column(String(16), default="processing")   # processing / ready / failed
    error_message = Column(Text, default="")            # status=failed 时的原因
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
