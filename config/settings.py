"""
企业级 RAG Agent 系统配置
基于 Pydantic Settings，支持 .env 文件加载和环境变量覆盖
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """全局配置单例"""

    # ===== LLM API =====
    llm_api_base: str = Field(default="http://localhost:8000/v1", alias="LLM_API_BASE")
    llm_api_key: str = Field(default="your-api-key-here", alias="LLM_API_KEY")
    llm_model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="LLM_MODEL_NAME")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # ===== Embedding =====
    embedding_model_name: str = Field(default="BAAI/bge-large-zh-v1.5", alias="EMBEDDING_MODEL_NAME")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_use_api: bool = Field(default=True, alias="EMBEDDING_USE_API")
    embedding_api_base: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="EMBEDDING_API_BASE")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_api_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_API_MODEL")
    embedding_api_dimension: int = Field(default=1024, alias="EMBEDDING_API_DIMENSION")
    embedding_api_batch_size: int = Field(default=10, alias="EMBEDDING_API_BATCH_SIZE")

    # ===== Reranker =====
    reranker_model_name: str = Field(default="BAAI/bge-reranker-large", alias="RERANKER_MODEL_NAME")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")

    # ===== Redis =====
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_session_ttl: int = Field(default=1800, alias="REDIS_SESSION_TTL")

    # ===== 数据库 =====
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/rag_agent.db",
        alias="DATABASE_URL",
    )

    # ===== 鉴权 (JWT) =====
    jwt_secret: str = Field(default="dev-insecure-secret-change-me-in-production-please", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60 * 24 * 7, alias="JWT_EXPIRE_MINUTES")  # 7 天

    # ===== 检索配置 =====
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K")
    faiss_top_k: int = Field(default=20, alias="FAISS_TOP_K")
    rerank_top_k: int = Field(default=10, alias="RERANK_TOP_K")
    merge_top_k: int = Field(default=30, alias="MERGE_TOP_K")
    min_relevance_score: float = Field(default=0.5, alias="MIN_RELEVANCE_SCORE")

    # ===== 自省配置 =====
    max_reflection_rounds: int = Field(default=2, alias="MAX_REFLECTION_ROUNDS")
    fact_check_confidence_threshold: float = Field(default=0.7, alias="FACT_CHECK_CONFIDENCE_THRESHOLD")

    # ===== 服务配置 =====
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ===== 路径配置 =====
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss_index"

    @property
    def bm25_corpus_path(self) -> Path:
        return self.data_dir / "bm25_corpus.json"

    @property
    def eval_dataset_path(self) -> Path:
        return self.data_dir / "eval_dataset.json"

    @property
    def knowledge_base_path(self) -> Path:
        return self.data_dir / "knowledge_base" / "sample_docs.json"

    @property
    def kb_indexes_dir(self) -> Path:
        """每个知识库独立索引的根目录（多租户物理隔离）"""
        return self.data_dir / "kb_indexes"

    def kb_index_dir(self, kb_id: str) -> Path:
        """指定知识库的索引目录：data/kb_indexes/{kb_id}/"""
        return self.kb_indexes_dir / kb_id

    @property
    def uploads_dir(self) -> Path:
        """上传文件的原始存储目录"""
        return self.project_root / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    def resolve_model(self, name: str) -> str:
        """
        智能解析模型路径：
        1. 如果 name 已经是本地路径且存在 → 直接用
        2. 如果 models/{name} 存在 → 返回本地路径（从 ModelScope 下载过）
        3. 否则 → 返回 HuggingFace ID（首次运行会自动下载）
        """
        local_path = self.models_dir / name
        if local_path.exists():
            return str(local_path)
        # 检查直接路径
        direct_path = Path(name)
        if direct_path.exists():
            return str(direct_path)
        return name

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# 全局单例
settings = Settings()
