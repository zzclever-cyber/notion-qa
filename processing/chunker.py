"""
句级切片 — 把长文本按句边界切成检索 chunk

产出的 chunk 结构与现有检索链路完全对齐（BM25 / FAISS / RRF 均以这些字段工作）：
    {"id", "title", "category", "content", "doc_id"}
- id:       "{doc_id}::c{i}"，全局唯一，删除文档时按此前缀清理
- content:  chunk 正文（用于 embedding / BM25 / 拼接上下文）
- title:    源文件名（引用来源展示用）
- category: 知识库名或 "uploaded"
- doc_id:   所属文档 id（额外字段，检索器会忽略，仅用于按文档删除）
"""
import re
from typing import List

# 句子终止符：中英文标点 + 换行。保留分隔符，尽量对齐句边界。
_SENT_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_PATTERN.findall(text) if s and s.strip()]


def chunk_document(
    text: str,
    doc_id: str,
    title: str,
    category: str = "uploaded",
    max_chars: int = 500,
    overlap_sentences: int = 1,
) -> List[dict]:
    """
    句级贪心切片：累积句子直到接近 max_chars，保证不切断句子；
    相邻 chunk 之间保留 overlap_sentences 句重叠，避免跨句语义丢失。

    Args:
        text: 已解析的纯文本
        doc_id: 文档 id（chunk id 前缀）
        title: 源文件名
        category: 分类标签（一般传知识库名）
        max_chars: 单个 chunk 目标最大字符数
        overlap_sentences: 相邻 chunk 的重叠句数
    Returns:
        chunk dict 列表；空文本返回 []
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for sent in sentences:
        # 单句就超长：先冲刷已有缓冲，再把长句按 max_chars 硬切
        if len(sent) > max_chars:
            if buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i:i + max_chars])
            continue

        if buf_len + len(sent) > max_chars and buf:
            chunks.append("".join(buf))
            # 保留末尾若干句作为重叠
            tail = buf[-overlap_sentences:] if overlap_sentences > 0 else []
            buf = list(tail)
            buf_len = sum(len(s) for s in buf)

        buf.append(sent)
        buf_len += len(sent)

    if buf:
        chunks.append("".join(buf))

    return [
        {
            "id": f"{doc_id}::c{i}",
            "title": title,
            "category": category,
            "content": chunk_text,
            "doc_id": doc_id,
        }
        for i, chunk_text in enumerate(chunks)
        if chunk_text.strip()
    ]
