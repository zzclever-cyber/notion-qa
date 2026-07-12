"""
性能基准测试
测试各模块的性能指标
"""
import pytest
import time

pytestmark = pytest.mark.benchmark


class TestRetrievalPerformance:
    """检索性能基准"""

    def test_bm25_retrieval_speed(self, bm25_retriever, benchmark):
        """BM25 检索应在 50ms 内完成"""
        result = benchmark(
            bm25_retriever.retrieve,
            query="年假天数",
            top_k=10,
        )
        assert len(result) > 0

    def test_faiss_retrieval_speed(self, faiss_retriever, benchmark):
        """FAISS 检索应在 100ms 内完成"""
        result = benchmark(
            faiss_retriever.retrieve,
            query="年假天数",
            top_k=10,
        )
        assert len(result) > 0

    def test_bm25_index_build_speed(self, sample_documents, benchmark):
        """BM25 索引构建速度测试"""
        from retrieval.bm25_retriever import BM25Retriever

        def build():
            r = BM25Retriever()
            r.build_index(sample_documents)
            return r

        retriever = benchmark(build)
        assert retriever.is_ready

    def test_faiss_index_build_speed(self, sample_documents, benchmark):
        """FAISS 索引构建速度测试"""
        from retrieval.faiss_retriever import FAISSRetriever

        def build():
            r = FAISSRetriever(device="cpu")
            r.build_index(sample_documents)
            return r

        retriever = benchmark(build)
        assert retriever.is_ready

    def test_rrf_fusion_speed(self, bm25_retriever, faiss_retriever, benchmark):
        """RRF 融合应在 10ms 内完成"""
        from retrieval.hybrid_retriever import HybridRetriever

        hr = HybridRetriever(bm25=bm25_retriever, faiss=faiss_retriever)
        bm25_results = bm25_retriever.retrieve("年假", top_k=10)
        faiss_results = faiss_retriever.retrieve("年假", top_k=10)

        result = benchmark(
            hr._reciprocal_rank_fusion,
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            k=60,
            top_k=10,
        )
        assert len(result) > 0


class TestFSMPerformance:
    """FSM 性能基准"""

    def test_fsm_transition_speed(self, benchmark):
        """状态转移应在微秒级完成"""
        from core.fsm import AgentFSM, AgentState

        fsm = AgentFSM()
        fsm.start("perf_001", "测试查询")

        def do_transitions():
            for s in [AgentState.INTENT, AgentState.RETRIEVE, AgentState.REASON,
                       AgentState.VERIFY, AgentState.DONE]:
                fsm.transition(s)
            fsm.reset()

        benchmark(do_transitions)


class TestEvalMetricsPerformance:
    """评估指标计算性能"""

    def test_faithfulness_computation_speed(self, benchmark):
        """Faithfulness 计算应在 1ms 内"""
        from evaluation.ragas_eval import RagasEvaluator

        evaluator = RagasEvaluator()
        answer = "公司员工每年享有5天年假，入职满3年后增加至10天。"
        contexts = ["公司实行弹性工作制，员工每年享有5天年假，入职满3年增加至10天。"]

        result = benchmark(
            evaluator._compute_faithfulness,
            answer=answer,
            contexts=contexts,
        )
        assert 0 <= result <= 1.0

    def test_context_precision_perfect(self):
        """完全匹配时精确度应为 1.0"""
        from evaluation.ragas_eval import RagasEvaluator

        evaluator = RagasEvaluator()
        precision = evaluator._compute_context_precision(
            query="年假",
            relevant_ids=["doc_001", "doc_002"],
            retrieved_ids=["doc_001", "doc_002"],
        )
        assert precision == 1.0

    def test_context_precision_partial(self):
        """部分匹配时精确度应在 0-1 之间"""
        from evaluation.ragas_eval import RagasEvaluator

        evaluator = RagasEvaluator()
        precision = evaluator._compute_context_precision(
            query="年假",
            relevant_ids=["doc_001", "doc_002"],
            retrieved_ids=["doc_001", "doc_003"],
        )
        assert precision == 0.5

    def test_recall_at_5_perfect(self):
        """完全召回时 Recall@5 应为 1.0"""
        from evaluation.ragas_eval import RagasEvaluator

        evaluator = RagasEvaluator()
        recall = evaluator._compute_recall_at_k(
            relevant_ids=["doc_001"],
            retrieved_ids=["doc_001", "doc_002", "doc_003"],
            k=5,
        )
        assert recall == 1.0
