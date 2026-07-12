"""
测试配置与共享 Fixtures
"""
import sys
import pytest
import asyncio
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """会话级事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_documents():
    """示例文档列表"""
    return [
        {
            "id": "doc_001",
            "title": "考勤管理制度",
            "category": "HR",
            "content": "公司实行弹性工作制，核心工作时间10:00-16:00。员工每年享有5天年假，入职满3年增加至10天。"
        },
        {
            "id": "doc_002",
            "title": "薪酬福利制度",
            "category": "HR",
            "content": "员工薪酬由基本工资（60%）、绩效工资（20%）、年终奖金（20%）组成。公司为员工缴纳五险一金。"
        },
        {
            "id": "doc_003",
            "title": "数据安全规范",
            "category": "IT",
            "content": "数据分为L1（公开）、L2（内部）、L3（机密）三个等级。L3数据必须使用AES-256加密存储。"
        },
    ]


@pytest.fixture
def sample_context():
    """示例检索上下文"""
    return (
        "【文档1】来源: 考勤管理制度 (分类: HR)\n"
        "公司实行弹性工作制，核心工作时间10:00-16:00。员工每年享有5天年假，入职满3年增加至10天。\n"
        "---\n"
        "【文档2】来源: 薪酬福利制度 (分类: HR)\n"
        "员工薪酬由基本工资（60%）、绩效工资（20%）、年终奖金（20%）组成。"
    )


@pytest.fixture
def sample_fsm_context():
    """示例 FSM 上下文"""
    from core.fsm import AgentContext, AgentState
    return AgentContext(
        session_id="test_session_001",
        query="公司年假有多少天？",
    )


@pytest.fixture
def bm25_retriever(sample_documents):
    """构建好的 BM25 检索器"""
    from retrieval.bm25_retriever import BM25Retriever
    r = BM25Retriever()
    r.build_index(sample_documents)
    return r


@pytest.fixture
def faiss_retriever(sample_documents):
    """构建好的 FAISS 检索器"""
    from retrieval.faiss_retriever import FAISSRetriever
    r = FAISSRetriever(device="cpu")
    r.build_index(sample_documents)
    return r


def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "benchmark: 性能基准测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
