"""
FAISS 稠密检索模块
支持两种 Embedding 模式：
1. 本地模型 — sentence-transformers 加载（需要先下载模型到 models/）
2. API 模式 — 调用 OpenAI 兼容的 Embedding API（无需下载模型，推荐国内使用）
"""
import json
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
import httpx
from config.settings import settings
from utils.logger import log


class FAISSRetriever:
    """FAISS 稠密向量检索器 — 支持本地模型和 API 两种模式（异步安全）"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        self.model = None            # SentenceTransformer 实例（仅本地模式）
        self.index: Optional[faiss.Index] = None
        self.documents: List[dict] = []
        self.doc_ids: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self._is_built = False
        self._dimension: int = 0
        self._use_api = settings.embedding_use_api
        self._http_client: Optional[httpx.AsyncClient] = None

    def _load_model(self):
        """延迟加载 embedding（本地模式）或验证 API 配置（API 模式）"""
        if self._use_api:
            log.info("[Embedding] 使用 API 模式，无需下载本地模型")
            # 先发一个测试请求获取维度
            self._dimension = settings.embedding_api_dimension
            return

        if self.model is None:
            from sentence_transformers import SentenceTransformer
            model_path = settings.resolve_model(self.model_name)
            log.info(f"[Embedding] 加载本地模型: {model_path}")
            self.model = SentenceTransformer(model_path, device=self.device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            log.info(f"[Embedding] 维度: {self._dimension}")

    async def _encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        编码文本为向量 — 自动选择本地（线程池）或 API（异步）模式
        """
        if self._use_api:
            return await self._encode_api(texts)
        # 本地模型 CPU 密集型，放入线程池避免阻塞事件循环
        return await asyncio.to_thread(
            self.model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=32,
        )

    async def _encode_api(self, texts: List[str]) -> np.ndarray:
        """
        调用 OpenAI 兼容 Embedding API 编码文本（纯异步，不阻塞事件循环）
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)

        all_embeddings = []
        batch_size = settings.embedding_api_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                api_base = settings.embedding_api_base or settings.llm_api_base
                api_key = settings.embedding_api_key or settings.llm_api_key
                resp = await self._http_client.post(
                    f"{api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.embedding_api_model,
                        "input": batch,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                resp_body = ""
                try:
                    resp_body = resp.text
                except Exception:
                    pass
                log.error(f"[Embedding API] 请求失败 (batch {i//batch_size}): {e}, 响应体: {resp_body}")
                raise

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # 归一化（使内积等价于余弦相似度）
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings

    async def aclose(self):
        """关闭 HTTP 客户端"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def build_index(self, documents: List[dict]):
        """构建 FAISS 向量索引"""
        log.info(f"开始构建 FAISS 索引，文档数: {len(documents)}, 模式: {'API' if self._use_api else '本地模型'}")
        self._load_model()
        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]

        log.info("正在生成文档嵌入向量...")
        self.doc_embeddings = await self._encode(contents, show_progress=not self._use_api)

        if self._dimension == 0:
            self._dimension = self.doc_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self._dimension)
        self.index.add(self.doc_embeddings.astype(np.float32))
        self._is_built = True
        log.info(f"FAISS 索引构建完成，包含 {self.index.ntotal} 个向量")

    async def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[dict, float]]:
        """检索最相关文档（异步安全）"""
        if not self._is_built:
            raise RuntimeError("FAISS 索引尚未构建，请先调用 build_index()")

        self._load_model()
        query_embedding = (await self._encode([query])).astype(np.float32)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results

    def save(self, index_path: Path):
        """保存 FAISS 索引和文档映射"""
        index_path.mkdir(parents=True, exist_ok=True)
        # 用 serialize_index + Python 文件 I/O 写盘：规避 faiss C++ 在含非 ASCII
        # (如中文)路径下 fopen 失败的问题（Windows GBK/ANSI 代码页无法打开 UTF-8 路径）
        idx_bytes = faiss.serialize_index(self.index)
        (index_path / "faiss.index").write_bytes(idx_bytes.tobytes())
        with open(index_path / "faiss_docs.json", "w", encoding="utf-8") as f:
            json.dump({"documents": self.documents}, f, ensure_ascii=False, indent=2)
        np.save(index_path / "faiss_embeddings.npy", self.doc_embeddings)
        log.info(f"FAISS 索引已保存至 {index_path}")

    def load(self, index_path: Path):
        """加载 FAISS 索引"""
        # 与 save() 对称：Python 读字节 + deserialize_index，兼容非 ASCII 路径
        idx_bytes = np.frombuffer((index_path / "faiss.index").read_bytes(), dtype=np.uint8)
        self.index = faiss.deserialize_index(idx_bytes)
        self._dimension = self.index.d
        with open(index_path / "faiss_docs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.doc_ids = [doc["id"] for doc in self.documents]
        embed_path = index_path / "faiss_embeddings.npy"
        if embed_path.exists():
            self.doc_embeddings = np.load(embed_path)
        self._is_built = True
        self._load_model()
        log.info(f"FAISS 索引已从 {index_path} 加载，共 {self.index.ntotal} 个向量")

    def get_document_by_id(self, doc_id: str) -> Optional[dict]:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None

    @property
    def is_ready(self) -> bool:
        return self._is_built

    @property
    def dimension(self) -> int:
        return self._dimension
