"""
Repository 模式
对数据库 CRUD 操作的封装，提供业务语义接口
"""
from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import (
    ChatRecord, EvalRecord, KnowledgeDoc,
    User, KnowledgeBase, Document,
)
from utils.logger import log


class ChatRepository:
    """对话记录仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, record: ChatRecord) -> ChatRecord:
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_by_session(self, session_id: str) -> Optional[ChatRecord]:
        result = await self.session.execute(
            select(ChatRecord).where(ChatRecord.session_id == session_id)
            .order_by(desc(ChatRecord.created_at)).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_history(
        self,
        session_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ChatRecord]:
        result = await self.session.execute(
            select(ChatRecord)
            .where(ChatRecord.session_id == session_id)
            .order_by(desc(ChatRecord.created_at))
            .limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def count_by_session(self, session_id: str) -> int:
        result = await self.session.execute(
            select(func.count()).where(ChatRecord.session_id == session_id)
        )
        return result.scalar()

    async def get_stats(self, days: int = 7) -> dict:
        """获取统计概览"""
        cutoff = datetime.utcnow()
        # Total records
        total_result = await self.session.execute(select(func.count()).select_from(ChatRecord))
        total = total_result.scalar()

        # Average metrics
        avg_result = await self.session.execute(
            select(
                func.avg(ChatRecord.faithfulness),
                func.avg(ChatRecord.answer_relevance),
                func.avg(ChatRecord.total_ms),
            )
        )
        row = avg_result.one()
        return {
            "total_conversations": total,
            "avg_faithfulness": round(row[0] or 0, 4),
            "avg_answer_relevance": round(row[1] or 0, 4),
            "avg_latency_ms": round(row[2] or 0, 1),
        }


class EvalRepository:
    """评估记录仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_batch(self, records: List[EvalRecord]) -> List[EvalRecord]:
        self.session.add_all(records)
        await self.session.commit()
        return records

    async def get_by_batch(self, batch_id: str) -> List[EvalRecord]:
        result = await self.session.execute(
            select(EvalRecord).where(EvalRecord.batch_id == batch_id)
        )
        return list(result.scalars().all())

    async def get_latest_batch_id(self) -> Optional[str]:
        result = await self.session.execute(
            select(EvalRecord.batch_id).order_by(desc(EvalRecord.created_at)).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_aggregate_by_type(self, batch_id: str) -> dict:
        """按查询类型汇总评估指标"""
        result = await self.session.execute(
            select(
                EvalRecord.query_type,
                func.count(),
                func.avg(EvalRecord.faithfulness),
                func.avg(EvalRecord.recall_at_5),
            ).where(EvalRecord.batch_id == batch_id)
            .group_by(EvalRecord.query_type)
        )
        return {
            row[0]: {"count": row[1], "avg_faithfulness": round(row[2] or 0, 4), "avg_recall_at_5": round(row[3] or 0, 4)}
            for row in result.all()
        }


class DocRepository:
    """知识库文档管理仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, doc: KnowledgeDoc) -> KnowledgeDoc:
        existing = await self.session.execute(
            select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc.doc_id)
        )
        existing_doc = existing.scalar_one_or_none()
        if existing_doc:
            existing_doc.content_hash = doc.content_hash
            existing_doc.content_length = doc.content_length
            existing_doc.updated_at = datetime.utcnow()
            existing_doc.version += 1
            await self.session.commit()
            return existing_doc
        else:
            self.session.add(doc)
            await self.session.commit()
            return doc

    async def get_by_id(self, doc_id: str) -> Optional[KnowledgeDoc]:
        result = await self.session.execute(
            select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id)
        )
        return result.scalar_one_or_none()

    async def list_active(self, limit: int = 100) -> List[KnowledgeDoc]:
        result = await self.session.execute(
            select(KnowledgeDoc)
            .where(KnowledgeDoc.is_active == True)
            .order_by(KnowledgeDoc.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def deactivate(self, doc_id: str) -> bool:
        doc = await self.get_by_id(doc_id)
        if doc:
            doc.is_active = False
            doc.updated_at = datetime.utcnow()
            await self.session.commit()
            return True
        return False


# ============================================================
# 多租户 SaaS 仓储 — 用户 / 知识库 / 文档
# ============================================================

class UserRepository:
    """用户仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_github_login(self, github_login: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.github_login == github_login)
        )
        return result.scalar_one_or_none()


class KnowledgeBaseRepository:
    """知识库仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, kb: KnowledgeBase) -> KnowledgeBase:
        self.session.add(kb)
        await self.session.commit()
        await self.session.refresh(kb)
        return kb

    async def get_by_id(self, kb_id: str) -> Optional[KnowledgeBase]:
        result = await self.session.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        )
        return result.scalar_one_or_none()

    async def list_by_user(self, user_id: str) -> List[KnowledgeBase]:
        result = await self.session.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.user_id == user_id)
            .order_by(desc(KnowledgeBase.created_at))
        )
        return list(result.scalars().all())

    async def delete(self, kb_id: str) -> bool:
        kb = await self.get_by_id(kb_id)
        if not kb:
            return False
        # 先删该 KB 下所有文档记录，再删 KB（索引文件由 service 层清理）
        await self.session.execute(
            delete(Document).where(Document.kb_id == kb_id)
        )
        await self.session.delete(kb)
        await self.session.commit()
        return True

    async def count_by_user(self, user_id: str) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(KnowledgeBase)
            .where(KnowledgeBase.user_id == user_id)
        )
        return result.scalar() or 0


class DocumentRepository:
    """文档仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, doc: Document) -> Document:
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_by_id(self, doc_id: str) -> Optional[Document]:
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def list_by_kb(self, kb_id: str) -> List[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.kb_id == kb_id)
            .order_by(desc(Document.created_at))
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: str = "",
    ) -> Optional[Document]:
        doc = await self.get_by_id(doc_id)
        if not doc:
            return None
        doc.status = status
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error_message:
            doc.error_message = error_message
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def delete(self, doc_id: str) -> bool:
        doc = await self.get_by_id(doc_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True

    async def count_by_kb(self, kb_id: str) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(Document)
            .where(Document.kb_id == kb_id)
        )
        return result.scalar() or 0

    async def list_kb_ids_for_user(self, user_id: str) -> List[str]:
        """该用户所有 KB 的 id（用于跨 KB 聚合统计）"""
        result = await self.session.execute(
            select(KnowledgeBase.id).where(KnowledgeBase.user_id == user_id)
        )
        return [row[0] for row in result.all()]
