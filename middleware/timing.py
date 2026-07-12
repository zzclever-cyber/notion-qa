"""
请求计时中间件
记录各阶段耗时并注入到响应头，用于性能分析和调优
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    请求耗时中间件
    注入 X-Response-Time-ms 响应头，用于客户端和网关的性能分析
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        elapsed_ms = int((time.time() - start) * 1000)

        response.headers["X-Response-Time-ms"] = str(elapsed_ms)
        response.headers["Server-Timing"] = f"total;dur={elapsed_ms}"

        return response
