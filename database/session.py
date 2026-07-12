"""
数据库会话管理
异步 SQLAlchemy 引擎 + 会话工厂
"""
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from config.settings import settings
from utils.logger import log


def _normalize_db_url(url: str) -> tuple[str, dict]:
    """
    规范化数据库 URL，让「直接粘贴 Neon 连接串」也能跑：
    - sqlite: 原样，附 check_same_thread
    - postgres/postgresql://  → postgresql+asyncpg://（异步驱动）
    - 剥离 asyncpg 不认识的查询参数（sslmode / channel_binding），
      SSL 需求改用 connect_args={"ssl": True} 传递
    返回 (规范化后的 url, connect_args)
    """
    if url.startswith("sqlite"):
        return url, {"check_same_thread": False}

    # 统一驱动
    if url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]

    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query))
    ssl_wanted = (
        q.pop("sslmode", "").lower() in {"require", "verify-ca", "verify-full"}
        or q.pop("ssl", "").lower() in {"true", "require"}
    )
    q.pop("channel_binding", None)  # asyncpg 不支持该参数
    new_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))
    connect_args = {"ssl": True} if (ssl_wanted or "neon.tech" in parts.netloc) else {}
    return new_url, connect_args


_db_url, _connect_args = _normalize_db_url(settings.database_url)
_is_sqlite = _db_url.startswith("sqlite")

# 异步引擎 — SQLite 不支持连接池；PostgreSQL 用默认池
engine = create_async_engine(
    _db_url,
    echo=False,
    poolclass=NullPool if _is_sqlite else None,
    connect_args=_connect_args,
)

# 异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """初始化数据库（创建所有表）"""
    from database.models import Base  # noqa: F811
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("数据库表已创建/已同步")


async def get_session() -> AsyncSession:
    """获取数据库会话（依赖注入）"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
