"""
知识库业务路由 — 知识库 CRUD / 文件上传 / 文档删除 / 用量统计

所有端点均需登录（Depends(get_current_user_id)），并校验资源归属当前用户。
上传后的解析+切片+建索引在 BackgroundTasks 中异步执行，文档状态：
    processing → ready / failed
"""
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form,
    BackgroundTasks, status,
)
from sqlalchemy import select, func

from api.schemas import (
    KnowledgeBaseCreate, KnowledgeBaseResponse, KnowledgeBaseDetail,
    DocumentResponse, UsageStatsResponse,
)
from config.settings import settings
from database.session import AsyncSessionLocal
from database.repository import KnowledgeBaseRepository, DocumentRepository
from database.models import KnowledgeBase, Document, ChatRecord
from middleware.auth import get_current_user_id
from processing.parser import detect_file_type, parse_file, SUPPORTED_TYPES
from processing.chunker import chunk_document
from processing.indexer import index_document, remove_document, delete_kb_index
from utils.logger import log

router = APIRouter(prefix="/api/v1", tags=["knowledge-base"])


# ============================================================
# 辅助
# ============================================================

def _doc_resp(d: Document) -> DocumentResponse:
    return DocumentResponse(
        id=d.id,
        kb_id=d.kb_id,
        filename=d.filename,
        file_type=d.file_type or "",
        chunk_count=d.chunk_count or 0,
        file_size_bytes=d.file_size_bytes or 0,
        status=d.status,
        error_message=d.error_message or "",
        created_at=d.created_at,
    )


async def _get_owned_kb(db, kb_id: str, user_id: str) -> KnowledgeBase:
    """加载知识库并校验归属；不存在或非本人一律 404（避免泄露存在性）"""
    kb = await KnowledgeBaseRepository(db).get_by_id(kb_id)
    if not kb or kb.user_id != user_id:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


def _invalidate_agent_cache(kb_id: str):
    """索引变更后失效 agent 的检索器缓存（延迟导入避免循环依赖）"""
    try:
        from main import get_agent_instance
        get_agent_instance().invalidate_retriever(kb_id)
    except Exception as e:
        log.debug(f"[kb] 失效检索器缓存跳过: {e}")


# ============================================================
# 知识库 CRUD
# ============================================================

@router.post("/knowledge-bases", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    req: KnowledgeBaseCreate,
    user_id: str = Depends(get_current_user_id),
):
    """创建知识库"""
    async with AsyncSessionLocal() as db:
        kb = await KnowledgeBaseRepository(db).create(KnowledgeBase(
            user_id=user_id,
            name=req.name,
            description=req.description or "",
        ))
        return KnowledgeBaseResponse(
            id=kb.id, name=kb.name, description=kb.description or "",
            document_count=0, created_at=kb.created_at,
        )


@router.get("/knowledge-bases", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(user_id: str = Depends(get_current_user_id)):
    """列出当前用户的所有知识库（含文档数）"""
    async with AsyncSessionLocal() as db:
        kbs = await KnowledgeBaseRepository(db).list_by_user(user_id)
        doc_repo = DocumentRepository(db)
        out = []
        for kb in kbs:
            cnt = await doc_repo.count_by_kb(kb.id)
            out.append(KnowledgeBaseResponse(
                id=kb.id, name=kb.name, description=kb.description or "",
                document_count=cnt, created_at=kb.created_at,
            ))
        return out


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseDetail)
async def get_knowledge_base(kb_id: str, user_id: str = Depends(get_current_user_id)):
    """知识库详情（含文档列表）"""
    async with AsyncSessionLocal() as db:
        kb = await _get_owned_kb(db, kb_id, user_id)
        docs = await DocumentRepository(db).list_by_kb(kb_id)
        return KnowledgeBaseDetail(
            id=kb.id, name=kb.name, description=kb.description or "",
            created_at=kb.created_at,
            documents=[_doc_resp(d) for d in docs],
        )


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(kb_id: str, user_id: str = Depends(get_current_user_id)):
    """删除知识库（连同文档记录 + 索引目录）"""
    async with AsyncSessionLocal() as db:
        await _get_owned_kb(db, kb_id, user_id)
        await KnowledgeBaseRepository(db).delete(kb_id)  # 删文档记录 + KB
    delete_kb_index(kb_id)                                # 删索引目录
    _invalidate_agent_cache(kb_id)
    return {"status": "deleted", "id": kb_id}


# ============================================================
# 文件上传 + 后台索引
# ============================================================

async def _process_document(
    doc_id: str, kb_id: str, file_path: str,
    file_type: str, filename: str, kb_name: str,
):
    """后台任务：解析 → 切片 → 建索引 → 更新文档状态"""
    try:
        text = parse_file(Path(file_path), file_type)
        chunks = chunk_document(text, doc_id=doc_id, title=filename, category=kb_name)
        if not chunks:
            raise ValueError("未从文件中解析出可索引文本")
        n = await index_document(kb_id, chunks)
        async with AsyncSessionLocal() as db:
            await DocumentRepository(db).update_status(doc_id, "ready", chunk_count=n)
        _invalidate_agent_cache(kb_id)
        log.info(f"[upload] 文档就绪 doc_id={doc_id} chunks={n}")
    except Exception as e:
        log.error(f"[upload] 文档处理失败 doc_id={doc_id}: {type(e).__name__}: {e}")
        async with AsyncSessionLocal() as db:
            await DocumentRepository(db).update_status(
                doc_id, "failed", error_message=str(e)[:500],
            )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background: BackgroundTasks,
    kb_id: str = Form(..., description="目标知识库ID"),
    file: UploadFile = File(..., description="PDF / Word / Markdown / txt"),
    user_id: str = Depends(get_current_user_id),
):
    """
    上传文档到指定知识库。
    立即返回 status=processing 的文档记录，后台异步完成解析+切片+建索引。
    """
    ftype = detect_file_type(file.filename or "")
    if ftype not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型，仅支持 {sorted(SUPPORTED_TYPES)}",
        )

    async with AsyncSessionLocal() as db:
        kb = await _get_owned_kb(db, kb_id, user_id)
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="文件为空")

        doc = await DocumentRepository(db).create(Document(
            kb_id=kb_id,
            filename=file.filename,
            file_type=ftype,
            file_size_bytes=len(content),
            status="processing",
        ))

        # 落盘 uploads/{user_id}/{doc_id}.{ext}
        dest_dir = settings.uploads_dir / user_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{doc.id}.{ftype}"
        dest.write_bytes(content)

        kb_name = kb.name
        resp = _doc_resp(doc)

    background.add_task(
        _process_document,
        doc_id=resp.id, kb_id=kb_id, file_path=str(dest),
        file_type=ftype, filename=file.filename, kb_name=kb_name,
    )
    log.info(f"[upload] 收到文件 {file.filename} → kb={kb_id} doc={resp.id}，已入后台队列")
    return resp


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str = Depends(get_current_user_id)):
    """删除文档并从所属知识库索引中移除其全部 chunk"""
    async with AsyncSessionLocal() as db:
        doc = await DocumentRepository(db).get_by_id(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")
        await _get_owned_kb(db, doc.kb_id, user_id)  # 校验归属
        kb_id = doc.kb_id
        await DocumentRepository(db).delete(doc_id)

    await remove_document(kb_id, doc_id)  # 重建索引（剔除该文档）
    _invalidate_agent_cache(kb_id)
    return {"status": "deleted", "id": doc_id}


# ============================================================
# 用量统计
# ============================================================

@router.get("/usage/stats", response_model=UsageStatsResponse)
async def usage_stats(user_id: str = Depends(get_current_user_id)):
    """当前用户用量：知识库数、文档数、问答次数、Token 消耗（估算）"""
    async with AsyncSessionLocal() as db:
        doc_repo = DocumentRepository(db)
        kb_ids = await doc_repo.list_kb_ids_for_user(user_id)
        doc_count = 0
        for kid in kb_ids:
            doc_count += await doc_repo.count_by_kb(kid)

        row = (await db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(ChatRecord.total_tokens), 0),
            ).where(ChatRecord.user_id == user_id)
        )).one()
        chat_count = int(row[0] or 0)
        total_tokens = int(row[1] or 0)

    return UsageStatsResponse(
        kb_count=len(kb_ids),
        document_count=doc_count,
        chat_count=chat_count,
        total_tokens=total_tokens,
    )
