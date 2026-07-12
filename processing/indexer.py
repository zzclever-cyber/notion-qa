"""
索引器 — 每个知识库维护一份独立的 FAISS + BM25 索引（多租户物理隔离）

目录布局:
    data/kb_indexes/{kb_id}/
        faiss_index/          # FAISSRetriever.save() 产物 (faiss.index / faiss_docs.json / *.npy)
        bm25_corpus.json      # BM25Retriever.save() 产物

设计取舍:
- 直接复用现有 FAISSRetriever / BM25Retriever，不改检索核心。
- FAISS 用的是 IndexFlatIP（暴力检索），Demo 规模下「整库重建」成本可忽略，
  因此新增/删除文档时都做全量重建，逻辑最简单、最不易出错。
  （大规模场景可改为增量 index.add + 缓存旧 embedding，此处按 KISS 处理。）
"""
import shutil
from pathlib import Path
from typing import List

from config.settings import settings
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from utils.logger import log


# ============================================================
# 路径（agent 加载 KB 检索器时复用这些函数，保证路径单一来源）
# ============================================================

def kb_faiss_dir(kb_id: str) -> Path:
    return settings.kb_index_dir(kb_id) / "faiss_index"


def kb_bm25_path(kb_id: str) -> Path:
    return settings.kb_index_dir(kb_id) / "bm25_corpus.json"


def kb_index_exists(kb_id: str) -> bool:
    """该 KB 是否已有可用索引"""
    return (kb_faiss_dir(kb_id) / "faiss.index").exists()


def _load_existing_docs(kb_id: str) -> List[dict]:
    """读取该 KB 现有的全部 chunk（从 faiss_docs.json）"""
    import json
    docs_file = kb_faiss_dir(kb_id) / "faiss_docs.json"
    if not docs_file.exists():
        return []
    try:
        with open(docs_file, "r", encoding="utf-8") as f:
            return json.load(f).get("documents", [])
    except Exception as e:
        log.warning(f"[indexer] 读取 KB {kb_id} 现有文档失败: {e}")
        return []


async def _rebuild(kb_id: str, all_docs: List[dict]):
    """用给定的全部 chunk 重建并保存该 KB 的 FAISS + BM25 索引"""
    if not all_docs:
        # 没有任何 chunk → 直接清空索引目录
        delete_kb_index(kb_id)
        return

    faiss = FAISSRetriever()
    try:
        await faiss.build_index(all_docs)
        faiss.save(kb_faiss_dir(kb_id))
    finally:
        await faiss.aclose()

    bm25 = BM25Retriever()
    bm25.build_index(all_docs)
    bm25.save(kb_bm25_path(kb_id))

    log.info(f"[indexer] KB {kb_id} 索引已重建，chunk 总数: {len(all_docs)}")


# ============================================================
# 对外接口
# ============================================================

async def index_document(kb_id: str, chunks: List[dict]) -> int:
    """
    把一个文档的 chunk 写入所属 KB 的索引（全量重建）。
    若同一 doc_id 已存在（重复上传），旧 chunk 会被替换。
    Returns:
        本次写入的 chunk 数
    """
    if not chunks:
        return 0
    doc_id = chunks[0].get("doc_id")
    existing = _load_existing_docs(kb_id)
    # 剔除同一文档的旧 chunk，避免重复
    existing = [
        d for d in existing
        if d.get("doc_id") != doc_id and not str(d.get("id", "")).startswith(f"{doc_id}::")
    ]
    all_docs = existing + chunks
    await _rebuild(kb_id, all_docs)
    return len(chunks)


async def remove_document(kb_id: str, doc_id: str) -> int:
    """
    从 KB 索引中移除某个文档的全部 chunk（全量重建）。
    Returns:
        移除后剩余的 chunk 数
    """
    existing = _load_existing_docs(kb_id)
    remaining = [
        d for d in existing
        if d.get("doc_id") != doc_id and not str(d.get("id", "")).startswith(f"{doc_id}::")
    ]
    await _rebuild(kb_id, remaining)
    return len(remaining)


def delete_kb_index(kb_id: str) -> None:
    """删除整个 KB 的索引目录（删除知识库时调用）"""
    kb_dir = settings.kb_index_dir(kb_id)
    if kb_dir.exists():
        shutil.rmtree(kb_dir, ignore_errors=True)
        log.info(f"[indexer] KB {kb_id} 索引目录已删除: {kb_dir}")
