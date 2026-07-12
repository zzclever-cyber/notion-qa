"""
Ragas 自动化评估集成模块
自动记录 RAG 三元组指标：
- Context Precision (上下文精确度)
- Faithfulness (答案忠实度)
- Answer Relevance (答案相关性)
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json
import asyncio
from pathlib import Path
from config.settings import settings
from utils.logger import log


@dataclass
class EvalMetrics:
    """单次评估指标"""
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    recall_at_5: float = 0.0

    def to_dict(self) -> dict:
        return {
            "context_precision": round(self.context_precision, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "recall_at_5": round(self.recall_at_5, 4),
        }


@dataclass
class EvalResult:
    """单次问答的完整评估记录"""
    session_id: str
    query: str
    expected_answer: str
    generated_answer: str
    retrieved_doc_ids: List[str]
    relevant_doc_ids: List[str]
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "relevant_doc_ids": self.relevant_doc_ids,
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp,
        }


class RagasEvaluator:
    """
    Ragas 自动化评估器
    封装 Ragas 指标计算逻辑，支持批量评估
    """

    def __init__(self):
        self.results: List[EvalResult] = []

    def evaluate_single(
        self,
        session_id: str,
        query: str,
        generated_answer: str,
        expected_answer: str,
        contexts: List[str],
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        fact_check_result: Optional[dict] = None,
    ) -> EvalResult:
        """
        评估单次问答
        Args:
            session_id: 会话ID
            query: 用户查询
            generated_answer: 生成的回答
            expected_answer: 期望回答
            contexts: 检索到的上下文字符串列表
            retrieved_doc_ids: 检索到的文档ID列表
            relevant_doc_ids: 标注的相关文档ID列表
            fact_check_result: LLM 事实核查结果（可选，优先用于 Faithfulness 计算）
        Returns:
            EvalResult
        """
        from datetime import datetime

        result = EvalResult(
            session_id=session_id,
            query=query,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            timestamp=datetime.now().isoformat(),
        )

        # 1. Faithfulness — LLM 事实核查优先，关键词兜底
        result.metrics.faithfulness = self._compute_faithfulness(
            generated_answer, contexts, fact_check_result
        )

        # 2. Answer Relevance — 答案相关性
        result.metrics.answer_relevance = self._compute_answer_relevance(
            query, generated_answer
        )

        # 3. Context Precision — 上下文精确度
        result.metrics.context_precision = self._compute_context_precision(
            query, relevant_doc_ids, retrieved_doc_ids
        )

        # 4. Recall@5 — 召回率
        result.metrics.recall_at_5 = self._compute_recall_at_k(
            relevant_doc_ids, retrieved_doc_ids, k=5
        )

        self.results.append(result)
        log.info(
            f"[Ragas] 评估完成: Faith={result.metrics.faithfulness:.3f}, "
            f"Relevance={result.metrics.answer_relevance:.3f}, "
            f"Precision={result.metrics.context_precision:.3f}, "
            f"Recall@5={result.metrics.recall_at_5:.3f}"
        )
        return result

    def evaluate_batch(
        self,
        samples: list,
    ) -> List[EvalResult]:
        """
        批量评估（同步执行）
        """
        self.results = []
        for sample in samples:
            try:
                self.evaluate_single(
                    session_id=f"eval_batch_{sample.id}",
                    query=sample.question,
                    generated_answer="",  # 需外部填充
                    expected_answer=sample.expected_answer,
                    contexts=[],
                    retrieved_doc_ids=[],
                    relevant_doc_ids=sample.relevant_doc_ids,
                )
            except Exception as e:
                log.error(f"评估样本 {sample.id} 失败: {e}")
        return self.results

    async def evaluate_batch_async(
        self,
        eval_items: list[dict],
    ) -> List[EvalResult]:
        """异步批量评估"""
        tasks = []
        for item in eval_items:
            tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self.evaluate_single,
                    item["session_id"],
                    item["query"],
                    item["generated_answer"],
                    item["expected_answer"],
                    item["contexts"],
                    item["retrieved_doc_ids"],
                    item["relevant_doc_ids"],
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        for r, item in zip(results, eval_items):
            if isinstance(r, Exception):
                log.error(f"异步评估失败 [{item['session_id']}]: {r}")
            else:
                valid.append(r)
        return valid

    # ============================================================
    # 指标计算方法
    # ============================================================

    def _compute_faithfulness(
        self,
        answer: str,
        contexts: List[str],
        fact_check_result: Optional[dict] = None,
    ) -> float:
        """
        计算答案忠实度（LLM 优先，关键词兜底）

        - 有 fact_check_result 时：基于 LLM 的声明级判决计算真实忠实度
          * verdict "consistent" → 1.0
          * verdict "partial" → max(0.3, 1.0 - contradiction_count * 0.25)
          * verdict "contradicted" → max(0.0, 1.0 - contradiction_count * 0.35)
        - 无 fact_check_result 时：回退到关键词匹配（快速但不精确）
        """
        # LLM 级判断（优先）
        if fact_check_result and isinstance(fact_check_result, dict):
            verdict = fact_check_result.get("verdict", "partial")
            contradiction_count = fact_check_result.get("contradiction_count", 0)
            claims = fact_check_result.get("claims", [])
            total_claims = len(claims) if claims else 1

            if verdict == "consistent":
                return 1.0
            elif verdict == "contradicted":
                return round(max(0.0, 1.0 - contradiction_count * 0.35), 4)
            else:  # partial
                return round(max(0.3, 1.0 - contradiction_count * 0.25), 4)

        # 关键词匹配兜底（无 LLM 事实核查结果时）
        if not answer or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        sentences = answer.replace("。", "\n").replace("；", "\n").split("\n")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if not sentences:
            return 0.0

        supported = 0
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            key_terms = [w for w in words if len(w) >= 3]
            if not key_terms:
                supported += 1
                continue
            match_count = sum(1 for t in key_terms if t.lower() in combined_context)
            if match_count >= len(key_terms) * 0.3:
                supported += 1

        return round(supported / len(sentences), 4)

    def _compute_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        计算答案相关性
        规则：基于查询和答案之间的词汇重叠度
        """
        if not query or not answer:
            return 0.0

        import jieba
        query_tokens = set(jieba.cut(query.lower()))
        answer_tokens = set(jieba.cut(answer.lower()))

        # 移除停用词（简化处理）
        stopwords = {"的", "是", "了", "在", "和", "有", "吗", "呢", "吧", "啊", "哦"}
        query_tokens = query_tokens - stopwords
        answer_tokens = answer_tokens - stopwords

        if not query_tokens:
            return 0.0

        overlap = len(query_tokens & answer_tokens)
        # Jaccard-like 相关性
        relevance = overlap / len(query_tokens)

        # 额外加分：答案包含数值（说明具体回答了问题）
        import re
        if re.search(r"\d+", answer):
            relevance = min(1.0, relevance + 0.1)

        return round(relevance, 4)

    def _compute_context_precision(
        self,
        query: str,
        relevant_ids: List[str],
        retrieved_ids: List[str],
    ) -> float:
        """
        计算上下文精确度
        Precision = |relevant ∩ retrieved| / |retrieved|
        """
        if not retrieved_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)
        precision = len(relevant_set & retrieved_set) / len(retrieved_set)
        return round(precision, 4)

    def _compute_recall_at_k(
        self,
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k: int = 5,
    ) -> float:
        """
        计算 Recall@K
        Recall@K = |relevant ∩ retrieved[:k]| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        top_k_set = set(retrieved_ids[:k])
        recall = len(relevant_set & top_k_set) / len(relevant_set)
        return round(recall, 4)

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """汇总所有评估结果的指标均值"""
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "avg_context_precision": round(
                sum(r.metrics.context_precision for r in self.results) / n, 4
            ),
            "avg_faithfulness": round(
                sum(r.metrics.faithfulness for r in self.results) / n, 4
            ),
            "avg_answer_relevance": round(
                sum(r.metrics.answer_relevance for r in self.results) / n, 4
            ),
            "avg_recall_at_5": round(
                sum(r.metrics.recall_at_5 for r in self.results) / n, 4
            ),
            "total_samples": n,
        }

    def save_results(self, path: Optional[Path] = None):
        """保存评估结果"""
        if path is None:
            path = settings.data_dir / "eval_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "aggregate": self.get_aggregate_metrics(),
                    "details": [r.to_dict() for r in self.results],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        log.info(f"评估结果已保存至 {path}")
