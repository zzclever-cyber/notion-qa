"""
限流中间件
基于滑动窗口的令牌桶算法，防止 API 滥用
"""
import time
import asyncio
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from utils.logger import log


class SlidingWindowRateLimiter:
    """
    滑动窗口限流器

    每个 IP 在 window_seconds 内最多允许 max_requests 次请求
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        clean_interval: float = 300.0,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clean_interval = clean_interval

        # {client_key: [timestamp, ...]}
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._last_clean = time.time()

    def _clean_expired(self):
        """清理过期的时间戳（防止内存泄漏）"""
        now = time.time()
        if now - self._last_clean < self.clean_interval and len(self._windows) < 100000:
            return
        cutoff = now - self.window_seconds
        expired_keys = []
        for key, timestamps in self._windows.items():
            self._windows[key] = [t for t in timestamps if t > cutoff]
            if not self._windows[key]:
                expired_keys.append(key)
        for key in expired_keys:
            del self._windows[key]
        # 安全阀：超过 100k 个 IP 记录时强制清理
        if len(self._windows) > 100000:
            sorted_keys = sorted(self._windows.keys(), key=lambda k: len(self._windows[k]))[:50000]
            for key in sorted_keys:
                del self._windows[key]
        self._last_clean = now

    def is_allowed(self, client_key: str) -> tuple[bool, int]:
        """
        检查是否允许请求
        Returns:
            (allowed, remaining)
        """
        self._clean_expired()
        now = time.time()
        cutoff = now - self.window_seconds

        timestamps = self._windows[client_key]
        # 移出窗口外的
        timestamps = [t for t in timestamps if t > cutoff]
        self._windows[client_key] = timestamps

        if len(timestamps) < self.max_requests:
            timestamps.append(now)
            return True, self.max_requests - len(timestamps)

        return False, 0

    def get_remaining(self, client_key: str) -> int:
        """获取剩余配额"""
        now = time.time()
        cutoff = now - self.window_seconds
        timestamps = [t for t in self._windows.get(client_key, []) if t > cutoff]
        return max(0, self.max_requests - len(timestamps))


# 全局限流器
_global_limiter = SlidingWindowRateLimiter(
    max_requests=120,   # 每分钟 120 次
    window_seconds=60.0,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    IP 级别限流中间件
    超出限制返回 429 Too Many Requests
    """

    def __init__(self, app, limiter: SlidingWindowRateLimiter = None):
        super().__init__(app)
        self.limiter = limiter or _global_limiter

    async def dispatch(self, request: Request, call_next):
        # 获取客户端标识
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP", "")
            or request.client.host if request.client else "unknown"
        )

        allowed, remaining = self.limiter.is_allowed(client_ip)

        if not allowed:
            log.warning(f"[限流] IP {client_ip} 超出速率限制")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"请求过于频繁，请稍后重试（限制: {self.limiter.max_requests}次/{self.limiter.window_seconds:.0f}秒）",
                    "retry_after": int(self.limiter.window_seconds),
                },
                headers={
                    "Retry-After": str(int(self.limiter.window_seconds)),
                    "X-RateLimit-Limit": str(self.limiter.max_requests),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time() + self.limiter.window_seconds)),
                },
            )

        response = await call_next(request)

        # 注入限流头
        response.headers["X-RateLimit-Limit"] = str(self.limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.limiter.window_seconds))

        return response


def get_global_limiter() -> SlidingWindowRateLimiter:
    return _global_limiter
