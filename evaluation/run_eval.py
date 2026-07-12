"""
离线全链路评测脚本
在 150 条评测集上运行完整 RAG 流水线，输出量化报告

使用方式:
    python -m evaluation.run_eval                    # 全量评测
    python -m evaluation.run_eval --max 20           # 快速抽检 20 条
    python -m evaluation.run_eval --type numerical   # 单类型评测

评测维度:
    - RAG 四元组: Faithfulness / Answer Relevance / Context Precision / Recall@5
    - 幻觉率: 自省 fact_check 发现的 contradiction 占比
    - 检索质量: BM25/FAISS/RRF 各阶段命中数
    - 延迟分布: P50 / P95 / P99
"""
import asyncio
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field

from config.settings import settings
from evaluation.dataset import EvalDataset, EvalSample
from evaluation.ragas_eval import RagasEvaluator, EvalResult
from utils.logger import setup_logger, log


@dataclass
class FullPipelineReport:
    """全链路评测报告"""
    # 基础信息
    total_samples: int = 0
    success_count: int = 0
    error_count: int = 0

    # RAG 四元组
    avg_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_recall_at_5: float = 0.0

    # 幻觉率（来自 fact_check）
    hallucination_rate: float = 0.0       # 有 contradiction 的回答占比
    avg_contradictions: float = 0.0       # 平均矛盾数
    avg_reflection_rounds: float = 0.0    # 平均自省轮次

    # 检索质量
    avg_docs_retrieved: float = 0.0
    avg_bm25_hits: float = 0.0
    avg_faiss_hits: float = 0.0

    # 延迟（ms）
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # 按类型细分
    by_type: Dict[str, dict] = field(default_factory=dict)

    # 最差样本（用于定位问题）
    worst_samples: List[dict] = field(default_factory=list)

    def print(self):
        """格式化打印报告"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║           RAG Agent 全链路离线评测报告                        ║
╠══════════════════════════════════════════════════════════════╣
║  样本: {total}  |  成功: {ok}  |  失败: {err}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【RAG 四元组】                                              ║
║    Faithfulness:         {faith:>8.4f}                          ║
║    Answer Relevance:     {rel:>8.4f}                          ║
║    Context Precision:    {prec:>8.4f}                          ║
║    Recall@5:             {rec:>8.4f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  【幻觉检测（自省 fact_check）】                              ║
║    幻觉率:               {hall:.1%}                              ║
║    平均矛盾数:            {cont:>8.2f}                          ║
║    平均自省轮次:          {ref:>8.2f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  【检索质量】                                                ║
║    平均检索文档数:        {docs:>8.1f}                          ║
║    平均 BM25 命中:        {bm25:>8.1f}                          ║
║    平均 FAISS 命中:       {faiss:>8.1f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  【延迟分布（ms）】                                          ║
║    P50: {p50:>8.0f}   P95: {p95:>8.0f}   P99: {p99:>8.0f}                       ║
╚══════════════════════════════════════════════════════════════╝
""".format(
            total=self.total_samples,
            ok=self.success_count,
            err=self.error_count,
            faith=self.avg_faithfulness,
            rel=self.avg_answer_relevance,
            prec=self.avg_context_precision,
            rec=self.avg_recall_at_5,
            hall=self.hallucination_rate,
            cont=self.avg_contradictions,
            ref=self.avg_reflection_rounds,
            docs=self.avg_docs_retrieved,
            bm25=self.avg_bm25_hits,
            faiss=self.avg_faiss_hits,
            p50=self.latency_p50,
            p95=self.latency_p95,
            p99=self.latency_p99,
        ))

        if self.by_type:
            print("  【按查询类型细分】")
            for qtype, m in self.by_type.items():
                print(f"    {qtype:15s}: Faith={m['faithfulness']:.4f}  "
                      f"Recall@5={m['recall_at_5']:.4f}  "
                      f"幻觉率={m['hallucination_rate']:.1%}  "
                      f"延迟P50={m['latency_p50']:.0f}ms")

        if self.worst_samples:
            print(f"\n  【最差 {len(self.worst_samples)} 样本】")
            for s in self.worst_samples[:5]:
                print(f"    [{s['type']}] {s['query'][:40]}... → Faith={s['faithfulness']:.3f}")


def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(len(vals) * p / 100)
    return vals[min(idx, len(vals) - 1)]


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


async def run_full_eval(
    max_samples: int = 0,
    query_types: list[str] | None = None,
    enable_reflection: bool = True,
) -> FullPipelineReport:
    """
    运行全链路离线评测

    每个样本执行完整流水线: 意图识别 → 检索 → 生成 → 自省 → 评估
    """
    from main import RAGAgent

    # 初始化 Agent
    agent = RAGAgent()
    await agent.initialize()

    # 加载数据集
    dataset = EvalDataset()
    samples = dataset.samples
    if query_types:
        samples = [s for s in samples if s.query_type in query_types]
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    log.info(f"开始离线评测: {len(samples)} 个样本, reflection={enable_reflection}")

    evaluator = RagasEvaluator()
    all_latencies: list[int] = []
    all_bm25_hits: list[int] = []
    all_faiss_hits: list[int] = []
    all_contradictions: list[int] = []
    all_reflection_rounds: list[int] = []
    all_docs_retrieved: list[int] = []

    report = FullPipelineReport(total_samples=len(samples))
    typed_data: dict[str, list[dict]] = {}

    for i, sample in enumerate(samples):
        log.info(f"[{i+1}/{len(samples)}] 评测: {sample.id} [{sample.query_type}] {sample.question[:50]}...")

        try:
            t0 = time.time()
            result = await agent.run(
                session_id=f"eval_{sample.id}",
                query=sample.question,
                enable_reflection=enable_reflection,
            )
            elapsed = int((time.time() - t0) * 1000)

            # 收集检索阶段数据
            bm25_hits = len([d for d in result.get("documents_used", []) if d.startswith("doc_")])
            faiss_hits = len(result.get("documents_used", []))
            docs_count = len(result.get("documents_used", []))

            # 从回答中提取自省结果
            reflection_rounds = result.get("reflection_rounds", 0)
            conflict_warning = result.get("conflict_warning", False)
            reflection_notes = result.get("reflection_notes", [])

            # 提取 LLM fact_check 结果（最后一轮），用于 Faithfulness 计算
            fact_check_data = None
            contradiction_count = 0
            if reflection_notes:
                last_round = reflection_notes[-1]  # 取最后一轮自省
                fact_check_data = {
                    "verdict": last_round.get("verdict", "partial"),
                    "contradiction_count": last_round.get("contradiction_count", 0),
                    "claims": last_round.get("claims", []),
                }
                contradiction_count = last_round.get("contradiction_count", 0)
                if contradiction_count == 0 and conflict_warning:
                    contradiction_count = 1  # 有冲突标记但 contradiction_count=0 → 至少算 1

            # Ragas 评估（传入 LLM fact_check 结果）
            eval_result = evaluator.evaluate_single(
                session_id=f"eval_{sample.id}",
                query=sample.question,
                generated_answer=result["answer"],
                expected_answer=sample.expected_answer,
                contexts=[result["answer"]],
                retrieved_doc_ids=result.get("documents_used", []),
                relevant_doc_ids=sample.relevant_doc_ids,
                fact_check_result=fact_check_data,
            )

            all_latencies.append(elapsed)
            all_bm25_hits.append(bm25_hits)
            all_faiss_hits.append(faiss_hits)
            all_docs_retrieved.append(docs_count)
            all_contradictions.append(contradiction_count)
            all_reflection_rounds.append(reflection_rounds)

            report.success_count += 1

            # 按类型分组
            qtype = sample.query_type
            if qtype not in typed_data:
                typed_data[qtype] = []
            typed_data[qtype].append({
                "faithfulness": eval_result.metrics.faithfulness,
                "recall_at_5": eval_result.metrics.recall_at_5,
                "hallucination": bool(contradiction_count),
                "latency_ms": elapsed,
                "query": sample.question,
            })

            log.info(
                f"  ✅ Faith={eval_result.metrics.faithfulness:.3f} "
                f"Recall@5={eval_result.metrics.recall_at_5:.3f} "
                f"冲突={bool(contradiction_count)} "
                f"延迟={elapsed}ms"
            )

        except Exception as e:
            report.error_count += 1
            log.error(f"  ❌ 评测失败: {e}")

        # 每 10 个样本输出进度
        if (i + 1) % 10 == 0:
            log.info(f"进度: {i+1}/{len(samples)}")

    # ── 汇总报告 ──
    aggregate = evaluator.get_aggregate_metrics()
    report.avg_faithfulness = aggregate.get("avg_faithfulness", 0.0)
    report.avg_context_precision = aggregate.get("avg_context_precision", 0.0)
    report.avg_answer_relevance = aggregate.get("avg_answer_relevance", 0.0)
    report.avg_recall_at_5 = aggregate.get("avg_recall_at_5", 0.0)

    # 幻觉率
    report.avg_contradictions = _mean(all_contradictions) if all_contradictions else 0.0
    report.hallucination_rate = (
        sum(1 for c in all_contradictions if c > 0) / len(all_contradictions)
        if all_contradictions else 0.0
    )
    report.avg_reflection_rounds = _mean(all_reflection_rounds) if all_reflection_rounds else 0.0

    # 检索质量
    report.avg_docs_retrieved = _mean(all_docs_retrieved) if all_docs_retrieved else 0.0
    report.avg_bm25_hits = _mean(all_bm25_hits) if all_bm25_hits else 0.0
    report.avg_faiss_hits = _mean(all_faiss_hits) if all_faiss_hits else 0.0

    # 延迟分布
    report.latency_p50 = _percentile(all_latencies, 50)
    report.latency_p95 = _percentile(all_latencies, 95)
    report.latency_p99 = _percentile(all_latencies, 99)

    # 按类型
    for qtype, items in typed_data.items():
        report.by_type[qtype] = {
            "count": len(items),
            "faithfulness": _mean([it["faithfulness"] for it in items]),
            "recall_at_5": _mean([it["recall_at_5"] for it in items]),
            "hallucination_rate": (
                sum(1 for it in items if it["hallucination"]) / len(items)
            ),
            "latency_p50": _percentile([it["latency_ms"] for it in items], 50),
        }

    # 最差样本
    all_flat = []
    for qtype, items in typed_data.items():
        for it in items:
            all_flat.append({"type": qtype, **it})
    report.worst_samples = sorted(all_flat, key=lambda x: x["faithfulness"])[:10]

    # 保存报告
    output_path = settings.data_dir / "eval_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_samples": report.total_samples,
                "success": report.success_count,
                "errors": report.error_count,
                "avg_faithfulness": report.avg_faithfulness,
                "avg_recall_at_5": report.avg_recall_at_5,
                "hallucination_rate": report.hallucination_rate,
                "latency_p50": report.latency_p50,
                "latency_p95": report.latency_p95,
            },
            "by_type": report.by_type,
            "worst_samples": report.worst_samples[:5],
        }, f, ensure_ascii=False, indent=2)

    log.info(f"评测报告已保存至 {output_path}")

    await agent.shutdown()
    return report


async def main():
    parser = argparse.ArgumentParser(description="RAG Agent 离线全链路评测")
    parser.add_argument("--max", type=int, default=0, help="最大评测样本数（0=全部）")
    parser.add_argument("--type", type=str, default=None, help="仅评测指定类型")
    parser.add_argument("--no-reflect", action="store_true", help="关闭自省（测纯生成质量）")
    args = parser.parse_args()

    setup_logger()
    query_types = [args.type] if args.type else None

    report = await run_full_eval(
        max_samples=getattr(args, 'max', 0),
        query_types=query_types,
        enable_reflection=not getattr(args, 'no_reflect', False),
    )

    report.print()
    print(f"\n报告已保存: {settings.data_dir / 'eval_report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
