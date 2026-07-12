"""
基线对比评测 — 量化不同策略的能力差异

对比维度:
    - 完整流水线 (BM25+FAISS+RRF+Reflection) — 当前最优方案
    - BM25-only               — 仅稀疏检索
    - FAISS-only              — 仅稠密检索
    - 无自省 (No-Reflection)   — 去掉 fact_check 纠正

输出: 每个维度的 Faithfulness / Recall@5 / 幻觉率 对比表

使用方式:
    python -m evaluation.baseline_compare --max 30
"""
import asyncio
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

from config.settings import settings
from evaluation.dataset import EvalDataset
from utils.logger import setup_logger, log


@dataclass
class BaselineResult:
    """单个策略的评测结果"""
    name: str
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_recall_at_5: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    sample_count: int = 0


def _mean(vals): return sum(vals) / len(vals) if vals else 0.0


async def evaluate_strategy(
    agent,
    samples: list,
    strategy: str,
    evaluator,
) -> BaselineResult:
    """
    使用指定策略评测所有样本
    strategy: "full" | "bm25_only" | "faiss_only" | "no_reflection"
    """
    result = BaselineResult(name=strategy, sample_count=len(samples))
    latencies = []
    faithfuls = []
    relevances = []
    recalls = []
    hallucinations = []

    for i, sample in enumerate(samples):
        try:
            enable_reflection = (strategy != "no_reflection")

            t0 = time.time()

            # 根据策略调整检索方式
            if strategy == "bm25_only":
                # 仅 BM25
                bm25_results = agent.hybrid_retriever.bm25.retrieve(sample.question, top_k=10)
                bm25_docs = [
                    type('RetrievedDocument', (), {
                        'doc': doc, 'doc_id': doc['id'],
                        'bm25_score': score, 'faiss_score': 0.0,
                        'rrf_score': score, 'rerank_score': 0.0, 'final_score': score,
                    })()
                    for doc, score in bm25_results
                ]
                ctx = agent.hybrid_retriever.get_context_for_llm(bm25_docs)
                answer = agent.llm.generate(
                    query=sample.question, context=ctx,
                    intent="factual", slot_params="{}",
                )
            elif strategy == "faiss_only":
                # 仅 FAISS
                faiss_results = await agent.hybrid_retriever.faiss.retrieve(sample.question, top_k=10)
                faiss_docs = [
                    type('RetrievedDocument', (), {
                        'doc': doc, 'doc_id': doc['id'],
                        'bm25_score': 0.0, 'faiss_score': score,
                        'rrf_score': score, 'rerank_score': 0.0, 'final_score': score,
                    })()
                    for doc, score in faiss_results
                ]
                ctx = agent.hybrid_retriever.get_context_for_llm(faiss_docs)
                answer = agent.llm.generate(
                    query=sample.question, context=ctx,
                    intent="factual", slot_params="{}",
                )
            else:
                # 完整流水线
                resp = await agent.run(
                    session_id=f"baseline_{strategy}_{sample.id}",
                    query=sample.question,
                    enable_reflection=enable_reflection,
                )
                answer = resp["answer"]

            elapsed = int((time.time() - t0) * 1000)
            latencies.append(elapsed)

            # 评估
            eval_r = evaluator.evaluate_single(
                session_id=f"baseline_{strategy}_{sample.id}",
                query=sample.question,
                generated_answer=answer,
                expected_answer=sample.expected_answer,
                contexts=[answer],
                retrieved_doc_ids=sample.relevant_doc_ids,
                relevant_doc_ids=sample.relevant_doc_ids,
            )
            faithfuls.append(eval_r.metrics.faithfulness)
            relevances.append(eval_r.metrics.answer_relevance)
            recalls.append(eval_r.metrics.recall_at_5)
            # 简化幻觉检测：检查回答是否包含"无法回答"/"无法确定"
            hallucinations.append(
                1 if any(w in answer for w in ["无法回答", "无法确定", "没有相关信息"]) else 0
            )

        except Exception as e:
            log.error(f"[{strategy}] 样本 {sample.id} 失败: {e}")
            continue

        if (i + 1) % 10 == 0:
            log.info(f"[{strategy}] 进度: {i+1}/{len(samples)}")

    n = len(latencies)
    result.avg_faithfulness = _mean(faithfuls)
    result.avg_answer_relevance = _mean(relevances)
    result.avg_recall_at_5 = _mean(recalls)
    result.hallucination_rate = _mean(hallucinations) if hallucinations else 0.0
    result.avg_latency_ms = _mean(latencies)
    result.success_rate = n / result.sample_count if result.sample_count else 0.0
    result.sample_count = n

    return result


async def main():
    parser = argparse.ArgumentParser(description="RAG Agent 基线对比评测")
    parser.add_argument("--max", type=int, default=30, help="最大评测样本数")
    args = parser.parse_args()

    setup_logger()

    from main import RAGAgent
    from evaluation.ragas_eval import RagasEvaluator

    agent = RAGAgent()
    await agent.initialize()

    dataset = EvalDataset()
    samples = dataset.samples[:getattr(args, 'max', 30)]

    evaluator = RagasEvaluator()

    strategies = [
        ("bm25_only", "BM25 仅稀疏检索"),
        ("faiss_only", "FAISS 仅稠密检索"),
        ("no_reflection", "混合检索 无自省"),
        ("full", "完整流水线 (BM25+FAISS+RRF+自省)"),
    ]

    results: list[BaselineResult] = []
    for strategy_key, strategy_label in strategies:
        log.info(f"\n{'='*50}\n开始评测: {strategy_label}\n{'='*50}")
        result = await evaluate_strategy(agent, samples, strategy_key, evaluator)
        result.name = strategy_label
        results.append(result)

    # ── 打印对比表 ──
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         RAG Agent 基线对比评测报告                                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  样本数: {results[0].sample_count}
╠══════════════════════════════════════════════════════════════════════════════════╣
║  {'策略':<40s} {'Faith':>8s} {'Recall@5':>10s} {'幻觉率':>8s} {'延迟':>8s} ║
╠══════════════════════════════════════════════════════════════════════════════════╣""")

    # 找到完整流水线作为基线
    full = next((r for r in results if "完整流水线" in r.name), results[-1])
    for r in results:
        faith_delta = f"+{r.avg_faithfulness - full.avg_faithfulness:+.3f}" if full.avg_faithfulness else ""
        marker = " ← 基线" if "完整流水线" in r.name else ""
        print(f"║  {r.name:<40s} {r.avg_faithfulness:>8.4f} {r.avg_recall_at_5:>10.4f} {r.hallucination_rate:>7.1%} {r.avg_latency_ms:>7.0f}ms{marker}")

    print("╚══════════════════════════════════════════════════════════════════════════════════╝")

    # 保存
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baselines": [
            {
                "name": r.name,
                "faithfulness": round(r.avg_faithfulness, 4),
                "recall_at_5": round(r.avg_recall_at_5, 4),
                "hallucination_rate": round(r.hallucination_rate, 4),
                "latency_ms": round(r.avg_latency_ms, 0),
                "success_rate": round(r.success_rate, 4),
            }
            for r in results
        ],
    }
    output_path = settings.data_dir / "baseline_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {output_path}")
    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
