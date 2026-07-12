"""
自定义评估指标与基准测试运行器
"""
from typing import List, Dict
from dataclasses import dataclass
from evaluation.ragas_eval import RagasEvaluator, EvalResult
from evaluation.dataset import EvalDataset, EvalSample
from utils.logger import log


@dataclass
class BenchmarkReport:
    """基准测试报告"""
    total_samples: int = 0
    avg_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_recall_at_5: float = 0.0
    by_query_type: Dict[str, dict] = None

    def __post_init__(self):
        if self.by_query_type is None:
            self.by_query_type = {}

    def print_report(self):
        """打印格式化的基准测试报告"""
        report = f"""
╔══════════════════════════════════════════════════════════╗
║              RAG Agent 基准测试报告                       ║
╠══════════════════════════════════════════════════════════╣
║  总样本数:           {self.total_samples:>6}                              ║
║  平均忠实度:          {self.avg_faithfulness:>.4f}                           ║
║  平均上下文精确度:    {self.avg_context_precision:>.4f}                           ║
║  平均答案相关性:      {self.avg_answer_relevance:>.4f}                           ║
║  平均 Recall@5:      {self.avg_recall_at_5:>.4f}                           ║
╠══════════════════════════════════════════════════════════╣
"""
        for qtype, metrics in self.by_query_type.items():
            report += f"║  {qtype:20s}: Faith={metrics['faithfulness']:.4f}, Recall@5={metrics['recall_at_5']:.4f} ║\n"
        report += "╚══════════════════════════════════════════════════════════╝"
        return report


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, evaluator: RagasEvaluator = None, dataset: EvalDataset = None):
        self.evaluator = evaluator or RagasEvaluator()
        self.dataset = dataset or EvalDataset()

    def run(self, answer_provider=None) -> BenchmarkReport:
        """
        运行完整基准测试
        Args:
            answer_provider: 可选，接受 query 返回 (answer, contexts, retrieved_doc_ids) 的回调
        Returns:
            BenchmarkReport
        """
        samples = self.dataset.samples
        if not samples:
            log.warning("数据集中没有样本")
            return BenchmarkReport()

        results = []
        for sample in samples:
            if answer_provider:
                answer, contexts, retrieved_ids = answer_provider(sample.question)
            else:
                # 无回调时使用预期答案作为生成答案（仅测试评估管道）
                answer = sample.expected_answer
                contexts = [sample.expected_answer]
                retrieved_ids = sample.relevant_doc_ids

            result = self.evaluator.evaluate_single(
                session_id=f"bench_{sample.id}",
                query=sample.question,
                generated_answer=answer,
                expected_answer=sample.expected_answer,
                contexts=contexts,
                retrieved_doc_ids=retrieved_ids,
                relevant_doc_ids=sample.relevant_doc_ids,
            )
            results.append(result)
            log.debug(f"[Benchmark] {sample.id}: Faith={result.metrics.faithfulness:.3f}")

        # 汇总
        report = BenchmarkReport(
            total_samples=len(results),
            avg_faithfulness=sum(r.metrics.faithfulness for r in results) / len(results),
            avg_context_precision=sum(r.metrics.context_precision for r in results) / len(results),
            avg_answer_relevance=sum(r.metrics.answer_relevance for r in results) / len(results),
            avg_recall_at_5=sum(r.metrics.recall_at_5 for r in results) / len(results),
        )

        # 按查询类型统计
        for qtype in ["single_hop", "multi_hop", "numerical", "negation"]:
            typed_results = [r for r, s in zip(results, samples) if s.query_type == qtype]
            if typed_results:
                report.by_query_type[qtype] = {
                    "count": len(typed_results),
                    "faithfulness": sum(r.metrics.faithfulness for r in typed_results) / len(typed_results),
                    "recall_at_5": sum(r.metrics.recall_at_5 for r in typed_results) / len(typed_results),
                }

        return report
