"""
BGE-Reranker 精排模块
使用 BGE-Reranker (Cross-Encoder) 对候选文档进行精细相关性排序

注意：FlagEmbedding（依赖 torch）改为延迟导入 —— torch 不可用时整个服务仍能启动，
精排阶段自动跳过（HybridRetriever 会检查 reranker.is_ready）。
"""
from typing import List, Tuple, Optional
from config.settings import settings
from utils.logger import log


class Reranker:
    """BGE Reranker 精排器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初始化精排器
        Args:
            model_name: reranker 模型名称
            device: 运行设备
        """
        self.model_name = model_name or settings.reranker_model_name
        self.device = device or settings.reranker_device
        self.model = None
        self._loaded = False

    def _load_model(self):
        """延迟加载模型（优先本地模型目录），依赖或模型不可用时降级跳过"""
        if self.model is None:
            model_path = settings.resolve_model(self.model_name)
            # 如果指向 HuggingFace ID（本地没有），尝试加载；失败则降级
            log.info(f"加载 Reranker 模型: {model_path}")
            try:
                from FlagEmbedding import FlagReranker  # 延迟导入：torch 不可用不影响服务启动
                self.model = FlagReranker(
                    model_path,
                    use_fp16=(self.device != "cpu"),
                    device=self.device,
                )
                self._loaded = True
            except Exception as e:
                log.warning(f"Reranker 模型加载失败，将跳过精排阶段: {e}")
                self._loaded = False
                self.model = None

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[dict, float]]:
        """
        对候选文档进行精排
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回数量
            score_threshold: 分数阈值，低于此值的文档被过滤
        Returns:
            [(文档dict, 相关性分数), ...]  按分数降序排列
        """
        if not documents:
            return []

        self._load_model()

        # 构建 (query, doc) 对
        pairs = [[query, doc["content"]] for doc in documents]

        # 批量计算相关性分数
        scores = self.model.compute_score(pairs, normalize=True)

        # 若单个文档，包装为列表
        if isinstance(scores, float):
            scores = [scores]

        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 过滤低分
        if score_threshold is not None:
            scored_docs = [
                (doc, score)
                for doc, score in scored_docs
                if score >= score_threshold
            ]

        return scored_docs[:top_k]

    @property
    def is_ready(self) -> bool:
        return self._loaded
