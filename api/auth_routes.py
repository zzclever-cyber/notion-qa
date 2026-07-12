"""
鉴权路由 — 注册 / 登录 / 当前用户 / GitHub OAuth 桥接

设计：前端 NextAuth 管理 GitHub 登录 session，登录成功后在 jwt callback 里
调用本模块的 /auth/oauth-callback，用 GitHub token 换后端 JWT。后端所有受保护
API 只认后端自己签发的 JWT（见 middleware/auth.py），与前端 session 解耦。
"""
import httpx
from fastapi import APIRouter, HTTPException, Depends, status

from api.schemas import (
    RegisterRequest, LoginRequest, TokenResponse, UserResponse,
    OAuthCallbackRequest,
)
from database.session import AsyncSessionLocal
from database.repository import UserRepository
from database.models import User
from middleware.auth import (
    hash_password, verify_password, create_access_token, get_current_user_id,
)
from utils.logger import log

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name or "",
        avatar_url=user.avatar_url or "",
    )


def _issue_token(user: User) -> TokenResponse:
    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user=_user_response(user))


# ============================================================
# 邮箱密码
# ============================================================

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """邮箱 + 密码注册，成功直接返回 JWT（免二次登录）"""
    email = req.email.strip().lower()
    async with AsyncSessionLocal() as db:
        repo = UserRepository(db)
        if await repo.get_by_email(email):
            raise HTTPException(status_code=409, detail="该邮箱已注册")
        user = User(
            email=email,
            hashed_password=hash_password(req.password),
            display_name=req.display_name or email.split("@")[0],
        )
        user = await repo.create(user)
    log.info(f"[auth] 新用户注册: {email}")
    return _issue_token(user)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """邮箱 + 密码登录"""
    email = req.email.strip().lower()
    async with AsyncSessionLocal() as db:
        repo = UserRepository(db)
        user = await repo.get_by_email(email)
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
    log.info(f"[auth] 用户登录: {email}")
    return _issue_token(user)


@router.get("/me", response_model=UserResponse)
async def me(user_id: str = Depends(get_current_user_id)):
    """获取当前登录用户信息"""
    async with AsyncSessionLocal() as db:
        user = await UserRepository(db).get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return _user_response(user)


# ============================================================
# GitHub OAuth 桥接
# ============================================================

async def _fetch_github_user(github_token: str) -> dict:
    """用 GitHub token 反查用户身份；失败返回空 dict（由调用方降级处理）"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            if resp.status_code == 200:
                return resp.json()
            log.warning(f"[auth] GitHub 用户校验失败: HTTP {resp.status_code}")
    except Exception as e:
        log.warning(f"[auth] GitHub 用户校验异常: {e}")
    return {}


@router.post("/oauth-callback", response_model=TokenResponse)
async def oauth_callback(req: OAuthCallbackRequest):
    """
    NextAuth GitHub 登录后回调：用 github_token 换后端 JWT。
    1. 用 token 反查 GitHub 身份（失败则信任前端传来的 github_user，便于本地演示）
    2. 按 github_login 查找/创建 User
    3. 签发后端 JWT
    """
    verified = await _fetch_github_user(req.github_token)
    gh = verified or req.github_user or {}

    login_name = gh.get("login")
    if not login_name:
        raise HTTPException(status_code=401, detail="无法确认 GitHub 身份")

    email = (gh.get("email") or f"{login_name}@users.noreply.github.com").strip().lower()
    avatar_url = gh.get("avatar_url", "") or ""
    display_name = gh.get("name") or login_name

    async with AsyncSessionLocal() as db:
        repo = UserRepository(db)
        user = await repo.get_by_github_login(login_name)
        if user is None:
            # 邮箱可能已被邮箱密码注册占用 → 复用同一账号并补 github_login
            user = await repo.get_by_email(email)
            if user is None:
                user = await repo.create(User(
                    email=email,
                    display_name=display_name,
                    avatar_url=avatar_url,
                    github_login=login_name,
                ))
            else:
                user.github_login = login_name
                user.avatar_url = avatar_url or user.avatar_url
                await db.commit()
                await db.refresh(user)

    log.info(f"[auth] GitHub 登录: {login_name} ({email})")
    return _issue_token(user)
