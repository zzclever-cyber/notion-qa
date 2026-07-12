"""
Prometheus 指标采集中间件
采集 HTTP 请求、RAG 流程、业务级别的可观测指标

指标清单：
- http_requests_total         — HTTP 请求总数（按 method/path/status 分标签）
- http_request_duration_ms    — HTTP 请求耗时分布
- rag_retrieval_duration_ms   — 检索阶段耗时
- rag_generation_duration_ms  — 生成阶段耗时
- rag_reflection_rounds       — 自省轮次分布
- rag_answers_total           — 回答总数（按意图类型分标签）
- rag_hallucination_score     — 幻觉评分（来自事实核查）
- llm_api_calls_total         — LLM API 调用次数
- llm_api_tokens_total        — LLM Token 消耗
- redis_connected             — Redis 连接状态
"""
import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from utils.logger import log


@dataclass
class MetricsRegistry:
    """
    内存指标注册表（asyncio 并发安全）

    设计说明：
    - asyncio 单线程协作式调度下，无 await 的操作天然原子
    - 使用 asyncio.Lock 保护 snapshot() 读写一致性
    - 生产环境可替换为 prometheus_client 库（多进程安全）
    """

    # HTTP 级别
    http_requests_total: Dict[str, int] = field(default_factory=dict)
    http_request_durations: list = field(default_factory=list)  # [(path, ms), ...]

    # RAG 流程级别
    rag_retrieval_durations: list = field(default_factory=list)
    rag_generation_durations: list = field(default_factory=list)
    rag_reflection_rounds: list = field(default_factory=list)
    rag_answers_total: Dict[str, int] = field(default_factory=dict)  # intent → count

    # LLM 级别
    llm_api_calls_total: int = 0
    llm_api_tokens_total: int = 0
    llm_api_errors_total: int = 0

    # 系统级别
    redis_connected: int = 0  # 0 或 1

    # 评估指标（最新值）
    avg_faithfulness: float = 0.0
    avg_recall_at_5: float = 0.0

    # 并发保护
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def record_http(self, method: str, path: str, status: int, duration_ms: float):
        key = f"{method}:{path}:{status}"
        self.http_requests_total[key] = self.http_requests_total.get(key, 0) + 1
        self.http_request_durations.append((path, duration_ms))
        # 防止列表无限增长
        if len(self.http_request_durations) > 10000:
            self.http_request_durations = self.http_request_durations[-5000:]

    def record_retrieval(self, duration_ms: float):
        self.rag_retrieval_durations.append(duration_ms)
        if len(self.rag_retrieval_durations) > 10000:
            self.rag_retrieval_durations = self.rag_retrieval_durations[-5000:]

    def record_generation(self, duration_ms: float):
        self.rag_generation_durations.append(duration_ms)
        if len(self.rag_generation_durations) > 10000:
            self.rag_generation_durations = self.rag_generation_durations[-5000:]

    def record_reflection(self, rounds: int):
        self.rag_reflection_rounds.append(rounds)
        if len(self.rag_reflection_rounds) > 10000:
            self.rag_reflection_rounds = self.rag_reflection_rounds[-5000:]

    def record_answer(self, intent: str):
        self.rag_answers_total[intent] = self.rag_answers_total.get(intent, 0) + 1

    def record_llm_call(self, tokens: int = 0):
        self.llm_api_calls_total += 1
        self.llm_api_tokens_total += tokens

    def record_llm_error(self):
        self.llm_api_errors_total += 1

    async def snapshot(self) -> dict:
        """生成 Prometheus 兼容的指标快照（加锁保证读写一致性）"""
        async with self._lock:
            recent_retrieval = (
                self.rag_retrieval_durations[-100:] if self.rag_retrieval_durations else []
            )
            recent_generation = (
                self.rag_generation_durations[-100:] if self.rag_generation_durations else []
            )

            return {
                # HTTP
                "http_requests_total": dict(self.http_requests_total),
                "http_request_p99_ms": _percentile(self.http_request_durations, 99) if self.http_request_durations else 0,
                "http_request_p50_ms": _percentile(self.http_request_durations, 50) if self.http_request_durations else 0,
                # RAG
                "rag_retrieval_avg_ms": round(_mean(recent_retrieval), 1) if recent_retrieval else 0,
                "rag_retrieval_p99_ms": round(_percentile(recent_retrieval, 99), 1) if recent_retrieval else 0,
                "rag_generation_avg_ms": round(_mean(recent_generation), 1) if recent_generation else 0,
                "rag_answers_by_intent": dict(self.rag_answers_total),
                "rag_reflection_avg_rounds": round(_mean(self.rag_reflection_rounds), 2) if self.rag_reflection_rounds else 0,
                # LLM
                "llm_api_calls_total": self.llm_api_calls_total,
                "llm_api_tokens_total": self.llm_api_tokens_total,
                "llm_api_errors_total": self.llm_api_errors_total,
                # System
                "redis_connected": self.redis_connected,
                # Eval
                "avg_faithfulness": self.avg_faithfulness,
                "avg_recall_at_5": self.avg_recall_at_5,
            }


# 全局单例
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _metrics


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(v[1] if isinstance(v, tuple) else v for v in values) / len(values)


def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(v[1] if isinstance(v, tuple) else v for v in values)
    idx = int(len(vals) * p / 100)
    return vals[min(idx, len(vals) - 1)]


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    HTTP 指标采集中间件
    自动记录每个请求的方法/路径/状态码和耗时
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()

        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = (time.time() - start) * 1000
            _metrics.record_http(request.method, request.url.path, 500, elapsed)
            raise

        elapsed = (time.time() - start) * 1000
        _metrics.record_http(request.method, request.url.path, response.status_code, elapsed)
        return response


def create_metrics_router():
    """创建 Prometheus 指标暴露路由"""
    from fastapi import APIRouter

    router = APIRouter(tags=["observability"])

    @router.get("/metrics")
    async def metrics():
        """Prometheus 兼容的指标端点"""
        snap = await _metrics.snapshot()

        lines = []
        # HTTP
        for key, count in snap["http_requests_total"].items():
            method, path, status = key.split(":", 2)
            lines.append(
                f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )
        lines.append(f"http_request_p99_ms {snap['http_request_p99_ms']}")
        lines.append(f"http_request_p50_ms {snap['http_request_p50_ms']}")

        # RAG
        lines.append(f"rag_retrieval_avg_ms {snap['rag_retrieval_avg_ms']}")
        lines.append(f"rag_retrieval_p99_ms {snap['rag_retrieval_p99_ms']}")
        lines.append(f"rag_generation_avg_ms {snap['rag_generation_avg_ms']}")
        lines.append(f"rag_reflection_avg_rounds {snap['rag_reflection_avg_rounds']}")

        for intent, count in snap["rag_answers_by_intent"].items():
            lines.append(f'rag_answers_total{{intent="{intent}"}} {count}')

        # LLM
        lines.append(f"llm_api_calls_total {snap['llm_api_calls_total']}")
        lines.append(f"llm_api_tokens_total {snap['llm_api_tokens_total']}")
        lines.append(f"llm_api_errors_total {snap['llm_api_errors_total']}")

        # System
        lines.append(f"redis_connected {snap['redis_connected']}")

        # Eval
        lines.append(f"avg_faithfulness {snap['avg_faithfulness']}")
        lines.append(f"avg_recall_at_5 {snap['avg_recall_at_5']}")

        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; charset=utf-8",
        )

    @router.get("/metrics/json")
    async def metrics_json():
        """JSON 格式的指标（便于调试和 Dashboard 消费）"""
        return await _metrics.snapshot()

    return router
