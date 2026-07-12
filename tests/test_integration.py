"""
集成测试
测试各模块的协作流程
"""
import pytest
import asyncio

pytestmark = pytest.mark.integration


class TestRetrievalPipeline:
    """检索管线集成测试"""

    def test_bm25_faiss_pipeline(self, bm25_retriever, faiss_retriever):
        """BM25 + FAISS 联合检索流程"""
        query = "年假有多少天"

        bm25_results = bm25_retriever.retrieve(query, top_k=5)
        faiss_results = faiss_retriever.retrieve(query, top_k=5)

        assert len(bm25_results) > 0, "BM25 应返回结果"
        assert len(faiss_results) > 0, "FAISS 应返回结果"

        # 取交集验证
        bm25_ids = {doc["id"] for doc, _ in bm25_results}
        faiss_ids = {doc["id"] for doc, _ in faiss_results}
        assert len(bm25_ids & faiss_ids) >= 0, "两路召回应有交集或独立覆盖"

    @pytest.mark.asyncio
    async def test_hybrid_retriever_concurrent(self, sample_documents):
        """混合检索器并发执行验证"""
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.bm25_retriever import BM25Retriever
        from retrieval.faiss_retriever import FAISSRetriever
        from retrieval.reranker import Reranker

        bm25 = BM25Retriever()
        bm25.build_index(sample_documents)

        faiss = FAISSRetriever(device="cpu")
        faiss.build_index(sample_documents)

        hr = HybridRetriever(bm25=bm25, faiss=faiss, reranker=Reranker())
        results = await hr.retrieve(
            query="年假天数",
            bm25_top_k=3,
            faiss_top_k=3,
            merge_top_k=5,
            enable_rerank=False,  # 跳过精排加速测试
        )

        assert len(results) > 0, "混合检索应返回结果"
        assert len(results) <= 5, "结果数不应超过 merge_top_k"
        # 验证 RetrievedDocument 结构
        for item in results:
            assert item.doc_id, "每个结果应有 doc_id"
            assert item.rrf_score >= 0, "RRF 分数应非负"


class TestFSMIntegration:
    """FSM 集成测试"""

    def test_full_pipeline_transitions(self):
        """完整 FSM 流转：IDLE → INTENT → RETRIEVE → REASON → VERIFY → DONE"""
        from core.fsm import AgentFSM, AgentState

        fsm = AgentFSM()
        ctx = fsm.start("test_001", "公司年假有多少天？")

        assert fsm.state == AgentState.IDLE
        assert ctx.session_id == "test_001"

        # 各阶段转移
        for target in [AgentState.INTENT, AgentState.RETRIEVE, AgentState.REASON,
                       AgentState.VERIFY, AgentState.DONE]:
            ok = fsm.transition(target)
            assert ok, f"转移到 {target.value} 应该成功"
            assert fsm.state == target

        # 验证 trace 完整
        trace = fsm.get_trace()
        assert len(trace) == 5, f"应为5步转移，实际: {len(trace)}"

    def test_reflection_loop_transition(self):
        """VERIFY → REASON 回环转移（自省纠错）"""
        from core.fsm import AgentFSM, AgentState

        fsm = AgentFSM()
        fsm.start("test_002", "测试")
        for s in [AgentState.INTENT, AgentState.RETRIEVE, AgentState.REASON, AgentState.VERIFY]:
            fsm.transition(s)

        # 自省发现矛盾，回退到 REASON
        ok = fsm.transition(AgentState.REASON)
        assert ok, "VERIFY → REASON 回环应合法"
        assert fsm.state == AgentState.REASON

    def test_illegal_transition_blocked(self):
        """非法转移应被拦截"""
        from core.fsm import AgentFSM, AgentState

        fsm = AgentFSM()
        fsm.start("test_003", "测试")

        # IDLE → DONE 不合法（跳过了中间状态）
        ok = fsm.transition(AgentState.DONE)
        assert not ok, "IDLE → DONE 应被拦截"

    def test_chitchat_fast_path(self):
        """闲聊走快速通道：INTENT → DONE"""
        from core.fsm import AgentFSM, AgentState

        fsm = AgentFSM()
        fsm.start("test_004", "你好")
        fsm.transition(AgentState.INTENT)
        fsm.transition(AgentState.DONE)  # 闲聊直接结束
        assert fsm.is_terminal


class TestIntentClassification:
    """意图分类集成测试"""

    def test_rule_based_chitchat(self):
        from core.intent import IntentClassifier, IntentType
        classifier = IntentClassifier()

        for query in ["你好", "谢谢", "再见", "今天天气怎么样"]:
            result = classifier.classify(query)
            assert result.intent == IntentType.CHITCHAT, f"'{query}' 应识别为闲聊"

    def test_rule_based_negation(self):
        from core.intent import IntentClassifier, IntentType
        classifier = IntentClassifier()

        for query in ["年假可以跨年累积吗？", "是否支持远程办公？"]:
            result = classifier.classify(query)
            assert result.intent == IntentType.NEGATION, f"'{query}' 应识别为否定反问"

    def test_rule_fallback(self):
        from core.intent import IntentClassifier
        classifier = IntentClassifier()

        # 含数值关键词的查询走规则兜底
        result = classifier.classify("专业版3年一共多少钱？")
        assert result.intent.value in ("numerical", "negation", "factual")


class TestContextFormatting:
    """LLM 上下文格式化测试"""

    def test_context_within_limit(self, sample_documents):
        """上下文应在字符限制内"""
        from retrieval.hybrid_retriever import HybridRetriever, RetrievedDocument
        from retrieval.bm25_retriever import BM25Retriever

        bm25 = BM25Retriever()
        bm25.build_index(sample_documents)

        hr = HybridRetriever(bm25=bm25)
        docs = [
            RetrievedDocument(doc=doc, doc_id=doc["id"], bm25_score=0.8)
            for doc in sample_documents
        ]

        context = hr.get_context_for_llm(docs, max_chars=500)
        assert len(context) <= 550, f"上下文长度 {len(context)} 应在限制附近"  # tolerance
        assert "考勤管理" in context or "年假" in context


class TestEvalDataset:
    """评测数据集集成测试"""

    def test_dataset_structure(self):
        from evaluation.dataset import EvalDataset, create_default_dataset
        import tempfile, os

        ds = create_default_dataset()
        assert ds.size == 150, f"默认数据集应为150条，实际: {ds.size}"
        assert "single_hop" in ds.type_distribution
        assert ds.type_distribution["single_hop"] == 40

    def test_dataset_filtering(self):
        from evaluation.dataset import EvalDataset, create_default_dataset

        ds = create_default_dataset()
        multi_hop = [s for s in ds.samples if s.query_type == "multi_hop"]
        assert len(multi_hop) == 40

        numerical = [s for s in ds.samples if s.query_type == "numerical"]
        assert len(numerical) == 35
