"""
Redis 会话存储模块
多用户并发下对话状态隔离：
- 每个会话绑定独立 namespace (rag:session:{session_id}:*)
- 支持 TTL 自动过期
- 避免内存存储的多用户串扰风险
- 关键操作带指数退避重试
"""
import json
import asyncio
from typing import Optional, Dict, Any, List
import redis.asyncio as aioredis
from config.settings import settings
from resilience.retry import retry_with_backoff
from utils.logger import log


class RedisSessionStore:
    """
    Redis 会话存储管理器
    使用 Redis Hash 存储每个会话的状态数据
    """

    NAMESPACE = "rag:session"
    DEFAULT_TTL = 1800  # 30分钟

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: Optional[int] = None,
    ):
        self.redis_url = redis_url or settings.redis_url
        self.ttl = ttl or settings.redis_session_ttl
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        """建立 Redis 连接（带重试）"""
        if self._client is None:
            self._client = await self._retry_redis_op(
                lambda: aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                ),
                "connect",
            )
            if self._client:
                try:
                    await self._retry_redis_op(
                        self._client.ping,
                        "ping",
                    )
                except Exception:
                    await self._client.close()
                    self._client = None
                    raise
            log.info(f"[Redis] 已连接: {self.redis_url}")

    async def disconnect(self):
        """断开连接"""
        if self._client:
            await self._client.close()
            self._client = None
            log.info("[Redis] 已断开连接")

    async def _retry_redis_op(self, op_factory, op_name: str, max_tries: int = 3, base_delay: float = 0.2):
        """
        Redis 操作重试包装器（工厂模式：每次重试调用 op_factory() 生成新协程）

        Args:
            op_factory: 无参可调用对象，返回一个 awaitable（协程）
            op_name: 操作名（日志用）
            max_tries: 最大尝试次数
            base_delay: 基础延迟（秒）
        """
        import random
        last_error = None
        for attempt in range(1, max_tries + 1):
            try:
                result = op_factory()
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                last_error = e
                if attempt == max_tries:
                    log.error(f"[Redis] {op_name} 重试{max_tries}次后仍然失败: {e}")
                    raise
                delay = base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())
                log.warning(f"[Redis] {op_name} 第{attempt}次失败, {delay:.2f}s后重试: {e}")
                await asyncio.sleep(delay)

    async def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> str:
        """
        创建新会话
        Args:
            session_id: 会话ID
            metadata: 初始元数据
        Returns:
            Redis key
        """
        key = self._session_key(session_id)
        data = {
            "session_id": session_id,
            "state": "idle",
            "created_at": str(await self._client.time()[0] if self._client else ""),
            "query": "",
            "intent": "",
            "answer": "",
            "reflection_rounds": "0",
            "eval_metrics": "{}",
        }
        if metadata:
            data["metadata"] = json.dumps(metadata, ensure_ascii=False)

        await self._client.hset(key, mapping=data)
        await self._client.expire(key, self.ttl)
        log.debug(f"[Redis] 创建会话: {key}")
        return key

    async def get_session(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        获取会话数据（带重试保护）
        """
        key = self._session_key(session_id)
        data = await self._retry_redis_op(
            lambda: self._client.hgetall(key),
            f"hgetall:{session_id}",
        )
        if not data:
            return None
        await self._retry_redis_op(
            lambda: self._client.expire(key, self.ttl),
            f"expire:{session_id}",
        )
        return data

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新会话字段（带重试保护）
        """
        key = self._session_key(session_id)
        serialized = {}
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                serialized[k] = json.dumps(v, ensure_ascii=False)
            else:
                serialized[k] = str(v)

        await self._retry_redis_op(
            lambda: self._client.hset(key, mapping=serialized),
            f"hset:{session_id}",
        )
        await self._retry_redis_op(
            lambda: self._client.expire(key, self.ttl),
            f"expire:{session_id}",
        )
        return True

    async def get_field(self, session_id: str, field: str) -> Optional[str]:
        """
        获取单个字段
        """
        key = self._session_key(session_id)
        value = await self._client.hget(key, field)
        await self._client.expire(key, self.ttl)
        return value

    async def set_field(self, session_id: str, field: str, value: Any) -> bool:
        """
        设置单个字段
        """
        key = self._session_key(session_id)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        await self._client.hset(key, field, value)
        await self._client.expire(key, self.ttl)
        return True

    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话（带重试保护）
        """
        key = self._session_key(session_id)
        deleted = await self._retry_redis_op(
            lambda: self._client.delete(key),
            f"delete:{session_id}",
        )
        return deleted > 0

    async def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        key = self._session_key(session_id)
        return await self._client.exists(key) > 0

    async def get_active_sessions(self) -> List[str]:
        """获取所有活跃会话ID"""
        pattern = f"{self.NAMESPACE}:*:state"
        keys = []
        async for key in self._client.scan_iter(match=pattern):
            namespace = key.decode("utf-8") if isinstance(key, bytes) else key
            # 提取 session_id
            parts = namespace.split(":")
            if len(parts) >= 3:
                keys.append(parts[2])
        return keys

    async def set_ttl(self, session_id: str, ttl: int):
        """更新会话TTL"""
        key = self._session_key(session_id)
        await self._client.expire(key, ttl)

    def _session_key(self, session_id: str) -> str:
        """构建 Redis key"""
        return f"{self.NAMESPACE}:{session_id}:data"

    @property
    def is_connected(self) -> bool:
        return self._client is not None
