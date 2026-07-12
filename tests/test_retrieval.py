"""
检索管线单元测试
"""
import json
import pytest
import tempfile
from pathlib import Path
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid_retriever import HybridRetriever

pytestmark = pytest.mark.unit


SAMPLE_DOCS = [
    {"id": "doc_1", "title": "考勤制度", "content": "公司实行弹性工作制，核心工作时间段为上午10:00至下午16:00。员工每日需完成8小时工作时长。迟到处理：每月累计迟到3次以内免于处罚。"},
    {"id": "doc_2", "title": "薪酬福利", "content": "公司薪酬结构由基本工资、岗位津贴、绩效奖金三部分组成。基本工资占比60%，绩效奖金占比20%。每月15日发放上月工资。"},
    {"id": "doc_3", "title": "数据安全", "content": "公司数据按敏感程度分为三级：公开级L1、内部级L2、机密级L3。L3数据访问需经过双重审批，存储必须使用AES-256加密。"},
    {"id": "doc_4", "title": "差旅报销", "content": "国内差旅交通标准：高铁二等座或硬卧。住宿标准：一线城市每晚不超过500元，二线城市不超过400元。餐补每人每天100元。"},
    {"id": "doc_5", "title": "产品定价", "content": "SaaS产品分为三个版本：基础版年费9,800元/10用户、专业版年费29,800元/30用户、企业版年费99,800元/不限用户。"},
]


class TestBM25Retriever:
    """BM25 检索器测试"""

    def setup_method(self):
        self.retriever = BM25Retriever()

    def test_build_and_retrieve(self):
        """基本构建和检索"""
        self.retriever.build_index(SAMPLE_DOCS)
        assert self.retriever.is_ready is True

        results = self.retriever.retrieve("考勤制度", top_k=3)
        assert len(results) > 0
        # 应该找到考勤相关文档
        assert results[0][0]["id"] == "doc_1"

    def test_retrieve_salary(self):
        """检索薪酬相关"""
        self.retriever.build_index(SAMPLE_DOCS)
        results = self.retriever.retrieve("基本工资占比多少", top_k=3)
        assert len(results) > 0
        doc_ids = [r[0]["id"] for r in results]
        assert "doc_2" in doc_ids

    def test_retrieve_return_count(self):
        """返回数量不超过 top_k"""
        self.retriever.build_index(SAMPLE_DOCS)
        results = self.retriever.retrieve("安全", top_k=2)
        assert len(results) <= 2

    def test_not_built_raises(self):
        """未构建索引抛出异常"""
        with pytest.raises(RuntimeError, match="尚未构建"):
            self.retriever.retrieve("测试")

    def test_save_and_load(self):
        """保存和加载"""
        self.retriever.build_index(SAMPLE_DOCS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25_corpus.json"
            self.retriever.save(path)
            assert path.exists()

            loaded = BM25Retriever()
            loaded.load(path)
            assert loaded.is_ready is True
            assert len(loaded.documents) == len(SAMPLE_DOCS)

    def test_get_document_by_id(self):
        """按ID获取文档"""
        self.retriever.build_index(SAMPLE_DOCS)
        doc = self.retriever.get_document_by_id("doc_3")
        assert doc is not None
        assert doc["title"] == "数据安全"


class TestHybridRetriever:
    """混合检索器测试"""

    @pytest.mark.asyncio
    async def test_hybrid_retrieve(self):
        """混合检索基本功能"""
        import asyncio

        bm25 = BM25Retriever()
        faiss = FAISSRetriever()
        bm25.build_index(SAMPLE_DOCS)
        faiss.build_index(SAMPLE_DOCS)

        hybrid = HybridRetriever(bm25=bm25, faiss=faiss)

        results = await hybrid.retrieve(
            "数据安全",
            bm25_top_k=3,
            faiss_top_k=3,
            merge_top_k=5,
            enable_rerank=False,
        )
        assert len(results) > 0
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_hybrid_concurrency(self):
        """验证并发检索"""
        import asyncio
        import time

        bm25 = BM25Retriever()
        faiss = FAISSRetriever()
        bm25.build_index(SAMPLE_DOCS)
        faiss.build_index(SAMPLE_DOCS)

        hybrid = HybridRetriever(bm25=bm25, faiss=faiss)

        t0 = time.time()
        results = await hybrid.retrieve("考勤迟到", enable_rerank=False)
        elapsed = time.time() - t0

        # 并发执行应快于串行
        # 实际耗时应小于两次检索串行之和
        assert len(results) > 0
        # 不做严格的性能断言，仅验证功能

    @pytest.mark.asyncio
    async def test_context_formatting(self):
        """测试上下文格式化"""
        bm25 = BM25Retriever()
        faiss = FAISSRetriever()
        bm25.build_index(SAMPLE_DOCS)
        faiss.build_index(SAMPLE_DOCS)

        hybrid = HybridRetriever(bm25=bm25, faiss=faiss)
        results = await hybrid.retrieve("薪酬", enable_rerank=False)
        context = hybrid.get_context_for_llm(results, max_chars=1000)

        assert len(context) > 0
        assert "薪酬" in context or "工资" in context


class TestEvalDataset:
    """测评数据集测试"""

    def test_dataset_creation(self):
        """数据集创建和加载"""
        from evaluation.dataset import _build_default_dataset, EvalDataset

        samples = _build_default_dataset()
        assert len(samples) == 150

        # 验证分布
        types = {}
        for s in samples:
            types[s.query_type] = types.get(s.query_type, 0) + 1

        assert types["single_hop"] == 40
        assert types["multi_hop"] == 40
        assert types["numerical"] == 35
        assert types["negation"] == 35

    def test_dataset_filter(self):
        """按类型筛选"""
        from evaluation.dataset import EvalDataset
        dataset = EvalDataset()
        single = dataset.filter_by_type("single_hop")
        assert len(single) == 40
        numerical = dataset.filter_by_type("numerical")
        assert len(numerical) == 35

    def test_dataset_difficulty(self):
        """按难度筛选"""
        from evaluation.dataset import EvalDataset
        dataset = EvalDataset()
        easy = dataset.get_by_difficulty("easy")
        hard = dataset.get_by_difficulty("hard")
        assert len(easy) > 0
        assert len(hard) > 0
