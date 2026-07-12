"""
JWT 鉴权模块

职责：
- 密码哈希 / 校验（bcrypt）
- 签发 / 解析后端 JWT（PyJWT，HS256）
- FastAPI 依赖 `get_current_user_id`：从 Authorization: Bearer <token> 提取 user_id

设计说明
--------
CLAUDE.md 里描述的是「中间件注入 request.state.user_id」，这里实现为 FastAPI 依赖
（Depends）而非全局中间件——依赖更易测试、能按路由精确控制、且天然跳过 auth/* 路由，
语义上等价于「注入 user_id」。原生 EventSource 无法自定义请求头，因此额外支持从
?token= 查询参数取 token（仅作流式端点的降级通道）。
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Request, HTTPException, status

from config.settings import settings
from utils.logger import log


# ============================================================
# 密码哈希
# ============================================================

# bcrypt 对超过 72 字节的密码会报错，统一截断
_BCRYPT_MAX_BYTES = 72


def _to_bcrypt_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """bcrypt 加盐哈希，返回可存库的字符串"""
    return bcrypt.hashpw(_to_bcrypt_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """校验明文密码与哈希是否匹配"""
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_to_bcrypt_bytes(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ============================================================
# JWT 签发 / 解析
# ============================================================

def create_access_token(user_id: str, expires_minutes: Optional[int] = None) -> str:
    """签发后端 access_token，payload 含 sub(user_id) + exp"""
    exp_minutes = expires_minutes or settings.jwt_expire_minutes
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=exp_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    """解析并校验 token，返回 payload；失败抛 jwt 异常"""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


# ============================================================
# FastAPI 依赖
# ============================================================

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="未认证或登录已过期，请重新登录",
    headers={"WWW-Authenticate": "Bearer"},
)


def _extract_token(request: Request) -> Optional[str]:
    """优先从 Authorization 头取 Bearer token，降级到 ?token= 查询参数"""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")


async def get_current_user_id(request: Request) -> str:
    """
    鉴权依赖：返回当前登录用户的 user_id（字符串 UUID）。
    在受保护路由上 `user_id: str = Depends(get_current_user_id)` 即可注入。
    """
    token = _extract_token(request)
    if not token:
        raise _UNAUTHORIZED
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise _UNAUTHORIZED
    except jwt.PyJWTError as e:
        log.warning(f"[auth] token 解析失败: {type(e).__name__}: {e}")
        raise _UNAUTHORIZED

    user_id = payload.get("sub")
    if not user_id:
        raise _UNAUTHORIZED
    # 同时写入 request.state，兼容 CLAUDE.md 里「注入 request.state.user_id」的语义
    request.state.user_id = user_id
    return user_id


async def get_optional_user_id(request: Request) -> Optional[str]:
    """
    可选鉴权：有合法 token 返回 user_id，否则返回 None（不抛异常）。
    用于聊天端点——登录用户走自己的知识库，匿名用户仍可访问默认全局库。
    """
    token = _extract_token(request)
    if not token:
        return None
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    if user_id:
        request.state.user_id = user_id
    return user_id
