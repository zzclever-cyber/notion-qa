"""
熔断器 (Circuit Breaker) 实现

状态机：CLOSED → OPEN → HALF_OPEN → CLOSED

使用场景：
- LLM API 调用保护（避免打爆下游）
- Redis 连接保护
- 任何可能雪崩的外部依赖
"""
import time
import asyncio
from enum import Enum
from functools import wraps
from dataclasses import dataclass
from utils.logger import log


class CircuitState(str, Enum):
    CLOSED = "closed"           # 正常通行
    OPEN = "open"               # 熔断打开，拒绝请求
    HALF_OPEN = "half_open"     # 半开，探测恢复


@dataclass
class CircuitStats:
    """熔断器统计"""
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_changes: list = None

    def __post_init__(self):
        self.state_changes = []


class CircuitBreaker:
    """
    熔断器

    参数：
        failure_threshold: 连续失败 N 次后熔断
        recovery_timeout: 熔断后等待 N 秒进入 HALF_OPEN
        half_open_max: HALF_OPEN 状态下允许的最大探测请求数
        consecutive_successes: HALF_OPEN 下连续成功 N 次才恢复
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 3,
        consecutive_successes: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.consecutive_successes = consecutive_successes

        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
        self.stats = CircuitStats()

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _transition_to(self, new_state: CircuitState):
        old = self.state
        self.state = new_state
        self.stats.state_changes.append({
            "from": old.value,
            "to": new_state.value,
            "time": time.time(),
        })
        log.warning(f"[熔断器:{self.name}] 状态变更: {old.value} → {new_state.value}")

    def _on_success(self):
        self.stats.total_calls += 1
        self.stats.total_successes += 1
        self.stats.last_success_time = time.time()
        self._failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_requests -= 1
            if self._success_count >= self.consecutive_successes:
                self._transition_to(CircuitState.CLOSED)
                self._success_count = 0

    def _on_failure(self):
        self.stats.total_calls += 1
        self.stats.total_failures += 1
        self.stats.last_failure_time = time.time()
        self._failure_count += 1

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            self._opened_at = time.time()
            self._success_count = 0
        elif (
            self.state == CircuitState.CLOSED
            and self._failure_count >= self.failure_threshold
        ):
            self._transition_to(CircuitState.OPEN)
            self._opened_at = time.time()

    def _before_request(self):
        """请求前检查，返回 True 表示允许通过"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                self._half_open_requests = 0
                self._success_count = 0
            else:
                raise CircuitBreakerOpenError(
                    f"熔断器[{self.name}]已打开，"
                    f"{self.recovery_timeout - (time.time() - self._opened_at):.0f}s 后重试"
                )

        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_requests >= self.half_open_max:
                raise CircuitBreakerOpenError(
                    f"熔断器[{self.name}]半开状态已达探测上限({self.half_open_max})"
                )
            self._half_open_requests += 1
            return True

        return True

    def call(self, func, *args, **kwargs):
        """同步调用保护"""
        self._before_request()
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception as e:
            self._on_failure()
            raise

    async def call_async(self, coro_func, *args, **kwargs):
        """异步调用保护"""
        self._before_request()
        try:
            result = await coro_func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception as e:
            self._on_failure()
            raise

    def decorator(self, func):
        """装饰器方式使用熔断器"""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
        return wrapper

    def reset(self):
        """手动重置熔断器"""
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        log.info(f"[熔断器:{self.name}] 手动重置")


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass


# ============================================================
# 预定义的熔断器实例
# ============================================================

# LLM API 熔断器
llm_circuit_breaker = CircuitBreaker(
    name="llm_api",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max=3,
    consecutive_successes=2,
)

# Redis 熔断器
redis_circuit_breaker = CircuitBreaker(
    name="redis",
    failure_threshold=3,
    recovery_timeout=15.0,
    half_open_max=2,
    consecutive_successes=1,
)
