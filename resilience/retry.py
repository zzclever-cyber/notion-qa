"""
重试模块
指数退避重试策略，利用 tenacity 库实现
支持同步/异步函数，可配置重试条件

使用示例：
    @retry_with_backoff(max_tries=3, base_delay=1.0)
    async def call_llm(prompt): ...
"""
import asyncio
import time
from functools import wraps
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Type
from utils.logger import log


@dataclass
class RetryConfig:
    """重试配置"""
    max_tries: int = 3
    base_delay: float = 1.0          # 基础延迟（秒）
    max_delay: float = 60.0          # 最大延迟（秒）
    exponential_base: float = 2.0    # 指数底数
    jitter: bool = True              # 是否添加随机抖动
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    no_retry_on_exceptions: Tuple[Type[Exception], ...] = ()  # 不重试的异常类型


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """计算指数退避延迟（带可选抖动）"""
    import random
    delay = min(config.base_delay * (config.exponential_base ** (attempt - 1)), config.max_delay)
    if config.jitter:
        delay = delay * (0.5 + random.random())
    return delay


def retry_with_backoff(
    max_tries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    no_retry_on: Tuple[Type[Exception], ...] = (),
):
    """
    重试装饰器（指数退避 + 抖动）

    Args:
        max_tries: 最大尝试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        retry_on: 需要重试的异常类型
        no_retry_on: 不重试的异常类型（优先级高于 retry_on）
    """
    config = RetryConfig(
        max_tries=max_tries,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on_exceptions=retry_on,
        no_retry_on_exceptions=no_retry_on,
    )

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, config.max_tries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except config.no_retry_on_exceptions:
                        raise
                    except config.retry_on_exceptions as e:
                        last_exception = e
                        if attempt == config.max_tries:
                            log.error(
                                f"[重试] {func.__name__} 已达最大重试次数({config.max_tries}): {e}"
                            )
                            raise
                        delay = _compute_delay(attempt, config)
                        log.warning(
                            f"[重试] {func.__name__} 第{attempt}次失败, "
                            f"{delay:.1f}s后重试: {e}"
                        )
                        await asyncio.sleep(delay)
                raise last_exception  # type: ignore
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, config.max_tries + 1):
                    try:
                        return func(*args, **kwargs)
                    except config.no_retry_on_exceptions:
                        raise
                    except config.retry_on_exceptions as e:
                        last_exception = e
                        if attempt == config.max_tries:
                            log.error(
                                f"[重试] {func.__name__} 已达最大重试次数({config.max_tries}): {e}"
                            )
                            raise
                        delay = _compute_delay(attempt, config)
                        log.warning(
                            f"[重试] {func.__name__} 第{attempt}次失败, "
                            f"{delay:.1f}s后重试: {e}"
                        )
                        time.sleep(delay)
                raise last_exception  # type: ignore
            return sync_wrapper
    return decorator


# ============================================================
# 预配置的重试装饰器
# ============================================================

# LLM API 调用重试（快速重试）
llm_retry = retry_with_backoff(
    max_tries=3,
    base_delay=0.5,
    max_delay=5.0,
)

# Redis 操作重试（较长间隔）
redis_retry = retry_with_backoff(
    max_tries=3,
    base_delay=0.2,
    max_delay=2.0,
)
