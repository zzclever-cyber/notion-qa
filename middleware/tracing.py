"""
请求追踪中间件
为每个请求生成唯一 Request ID，注入日志上下文，通过响应头传递
用于全链路追踪：API → FSM → 检索 → 生成 → 自省
"""
import uuid
import time
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from utils.logger import log


# ContextVar 用于在异步上下文中传递 request_id
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def get_request_id() -> str:
    """获取当前请求的 Request ID"""
    return request_id_var.get()


def get_session_id() -> str:
    """获取当前请求的 Session ID"""
    return session_id_var.get()


def set_session_id(sid: str):
    """设置当前请求的 Session ID（在路由处理中调用）"""
    session_id_var.set(sid)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    请求追踪中间件
    1. 为每个请求生成 X-Request-ID
    2. 注入到 ContextVar 供全链路使用
    3. 通过响应头返回给客户端
    4. 记录请求开始/结束日志
    """

    async def dispatch(self, request: Request, call_next):
        # 优先使用客户端传入的，否则生成新 ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4().hex[:16]))
        request_id_var.set(request_id)

        start_time = time.time()
        method = request.method
        path = request.url.path

        log.bind(request_id=request_id).info(
            f"[请求] {method} {path} 开始"
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((time.time() - start_time) * 1000)
            log.bind(request_id=request_id).error(
                f"[请求] {method} {path} 异常 ({elapsed_ms}ms): {exc}"
            )
            raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        status_code = response.status_code

        # 注入追踪头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = str(elapsed_ms)

        log.bind(request_id=request_id).info(
            f"[请求] {method} {path} → {status_code} ({elapsed_ms}ms)"
        )

        # 慢请求告警
        if elapsed_ms > 3000:
            log.bind(request_id=request_id).warning(
                f"[慢请求] {method} {path} 耗时 {elapsed_ms}ms (>3000ms)"
            )

        return response
