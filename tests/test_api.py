"""
API 集成测试
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.integration

# 在导入 app 前模拟依赖，避免实际连接 Redis/加载模型
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def test_app():
    """创建测试用的 FastAPI 应用"""
    from main import app, _agent_instance, RAGAgent

    # 创建 mock agent
    mock_agent = RAGAgent()
    mock_agent.initialize = AsyncMock()
    mock_agent.shutdown = AsyncMock()
    mock_agent.run = AsyncMock(return_value={
        "session_id": "test_sess",
        "query": "测试",
        "answer": "测试回答",
        "intent": "factual",
        "documents_used": ["doc_1", "doc_2"],
        "reflection_rounds": 1,
        "conflict_warning": False,
        "trace": [],
        "timings": {"total_ms": 100},
    })
    mock_agent.get_session = AsyncMock(return_value={
        "session_id": "test_sess",
        "state": "done",
        "query": "测试",
        "intent": "factual",
        "answer": "测试回答",
        "reflection_rounds": "1",
    })
    mock_agent.delete_session = AsyncMock(return_value=True)
    mock_agent.list_active_sessions = AsyncMock(return_value=["sess_1", "sess_2"])
    mock_agent.evaluate_batch = AsyncMock(return_value=[])
    mock_agent.hybrid_retriever = MagicMock()
    mock_agent.hybrid_retriever.bm25.is_ready = True
    mock_agent.hybrid_retriever.faiss.is_ready = True
    mock_agent.session_store = MagicMock()
    mock_agent.session_store.is_connected = True

    # 注入 mock agent
    import main
    main._agent_instance = mock_agent

    client = TestClient(app)
    return client


class TestHealthEndpoint:
    """健康检查端点"""

    def test_health(self, test_app):
        response = test_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "retrievers" in data


class TestChatEndpoint:
    """聊天端点"""

    def test_chat_basic(self, test_app):
        response = test_app.post("/chat", json={
            "query": "公司年假有多少天？",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] is not None
        assert "session_id" in data

    def test_chat_with_session(self, test_app):
        response = test_app.post("/chat", json={
            "query": "测试查询",
            "session_id": "my_session_123",
            "enable_reflection": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "my_session_123"

    def test_chat_empty_query(self, test_app):
        """空查询应该返回422"""
        response = test_app.post("/chat", json={"query": ""})
        assert response.status_code == 422

    def test_chat_query_too_long(self, test_app):
        """过长查询返回422"""
        response = test_app.post("/chat", json={"query": "x" * 3000})
        assert response.status_code == 422


class TestSessionEndpoints:
    """会话管理端点"""

    def test_get_session(self, test_app):
        response = test_app.get("/session/test_sess")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_sess"

    def test_delete_session(self, test_app):
        response = test_app.delete("/session/test_sess")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

    def test_list_sessions(self, test_app):
        response = test_app.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data
        assert data["count"] == 2


class TestEvalEndpoints:
    """评估端点"""

    def test_run_eval(self, test_app):
        response = test_app.post("/eval/run", json={})
        assert response.status_code == 200
        data = response.json()
        assert "total_samples" in data

    def test_get_eval_results_not_found(self, test_app):
        response = test_app.get("/eval/results")
        # 文件可能不存在
        assert response.status_code in (200, 404)
