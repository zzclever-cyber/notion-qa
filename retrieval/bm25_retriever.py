"""
BM25 稀疏检索模块
使用 rank_bm25 + jieba 中文分词实现关键词匹配检索
"""
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from utils.logger import log


class BM25Retriever:
    """BM25 稀疏检索器"""

    def __init__(self, corpus_path: Optional[Path] = None):
        """
        初始化检索器
        Args:
            corpus_path: 语料库JSON文件路径
        """
        self.corpus_path = corpus_path
        self.documents: List[dict] = []
        self.doc_ids: List[str] = []
        self.doc_contents: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_built = False

    def build_index(self, documents: List[dict]):
        """
        构建BM25索引
        Args:
            documents: 文档列表，每个文档包含 id, content 等字段
        """
        log.info(f"开始构建 BM25 索引，文档数: {len(documents)}")
        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]
        self.doc_contents = [doc["content"] for doc in documents]

        # 中文分词后构建BM25
        self.tokenized_corpus = [
            list(jieba.cut(content)) for content in self.doc_contents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._is_built = True
        log.info(f"BM25 索引构建完成，词表大小: {len(self.bm25.idf)}")

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[dict, float]]:
        """
        检索与查询最相关的文档
        Args:
            query: 查询文本
            top_k: 返回数量
        Returns:
            [(文档dict, BM25分数), ...]
        """
        if not self._is_built:
            raise RuntimeError("BM25 索引尚未构建，请先调用 build_index()")

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        # 取 top_k
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 仅返回有相关性的结果
                results.append((self.documents[idx], float(scores[idx])))

        return results[:top_k]

    def save(self, path: Path):
        """保存 BM25 索引到磁盘"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"BM25 语料已保存至 {path}")

    def load(self, path: Path):
        """从磁盘加载并重建 BM25 索引"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.build_index(data["documents"])
        log.info(f"BM25 索引已从 {path} 加载，共 {len(self.documents)} 篇文档")

    @property
    def is_ready(self) -> bool:
        return self._is_built

    def get_document_by_id(self, doc_id: str) -> Optional[dict]:
        """按ID获取文档"""
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
