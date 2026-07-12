"""
API 请求/响应 Pydantic 模型
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================
# 知识库 / 文档 / 用量
# ============================================================

class KnowledgeBaseCreate(BaseModel):
    """创建知识库"""
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = Field(None, max_length=1000)


class DocumentResponse(BaseModel):
    """文档信息"""
    id: str
    kb_id: str
    filename: str
    file_type: str = ""
    chunk_count: int = 0
    file_size_bytes: int = 0
    status: str = "processing"          # processing / ready / failed
    error_message: str = ""
    created_at: Optional[datetime] = None


class KnowledgeBaseResponse(BaseModel):
    """知识库概要（列表用）"""
    id: str
    name: str
    description: str = ""
    document_count: int = 0
    created_at: Optional[datetime] = None


class KnowledgeBaseDetail(BaseModel):
    """知识库详情（含文档列表）"""
    id: str
    name: str
    description: str = ""
    created_at: Optional[datetime] = None
    documents: List[DocumentResponse] = Field(default_factory=list)


class UsageStatsResponse(BaseModel):
    """用户用量统计"""
    kb_count: int = 0
    document_count: int = 0
    chat_count: int = 0
    total_tokens: int = 0


# ============================================================
# 鉴权 / 用户
# ============================================================

class RegisterRequest(BaseModel):
    """邮箱密码注册"""
    email: str = Field(..., description="邮箱", max_length=255)
    password: str = Field(..., min_length=6, max_length=128, description="密码（≥6位）")
    display_name: Optional[str] = Field(None, max_length=128, description="昵称，默认取邮箱前缀")


class LoginRequest(BaseModel):
    """邮箱密码登录"""
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=128)


class TokenResponse(BaseModel):
    """登录/注册返回的后端 JWT"""
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    """用户公开信息"""
    id: str
    email: str
    display_name: str = ""
    avatar_url: str = ""


class OAuthCallbackRequest(BaseModel):
    """NextAuth GitHub 登录后回调，用 github_token 换后端 JWT"""
    github_token: str = Field(..., description="GitHub OAuth access_token")
    github_user: Dict[str, Any] = Field(
        default_factory=dict,
        description="GitHub 用户信息 {login, email, avatar_url, name}",
    )


# ============================================================
# 聊天
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动创建")
    kb_id: Optional[str] = Field(None, description="知识库ID，指定则只在该库检索；不传用默认全局库")
    enable_reflection: bool = Field(True, description="是否启用自省机制")


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    query: str
    answer: str
    intent: str = ""
    documents_used: List[str] = Field(default_factory=list)
    reflection_rounds: int = 0
    conflict_warning: bool = False
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    timings: Dict[str, int] = Field(default_factory=dict)


class StreamChatResponse(BaseModel):
    """流式聊天响应（SSE）"""
    event: str
    data: str


# ============================================================
# 会话
# ============================================================

class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    state: str
    query: str = ""
    intent: str = ""
    answer: str = ""
    reflection_rounds: int = 0


class SessionListResponse(BaseModel):
    """会话列表（分页）"""
    active_sessions: List[str]
    count: int
    total: int = 0
    page: int = 1
    page_size: int = 50


class PaginatedResponse(BaseModel):
    """通用分页响应"""
    items: List[Any] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
    pages: int = 0


# ============================================================
# 评估
# ============================================================

class EvalRequest(BaseModel):
    """评估请求"""
    dataset_path: Optional[str] = None
    query_types: Optional[List[str]] = None
    max_samples: Optional[int] = None


class EvalResponse(BaseModel):
    """评估响应"""
    total_samples: int
    avg_faithfulness: float
    avg_context_precision: float
    avg_answer_relevance: float
    avg_recall_at_5: float
    by_query_type: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


# ============================================================
# 反馈闭环
# ============================================================

class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    session_id: str = Field(..., description="会话ID")
    rating: str = Field(..., description="thumbs_up / thumbs_down / neutral")
    comment: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = Field(None, description="反馈标签，如['事实错误','回答不完整','检索不相关']")


class FeedbackResponse(BaseModel):
    """反馈确认"""
    session_id: str
    status: str  # "recorded"
    message: str


# ============================================================
# 健康检查
# ============================================================

class HealthResponse(BaseModel):
    """健康检查"""
    status: str
    version: str = "1.0.0"
    retrievers: Dict[str, bool] = Field(default_factory=dict)
    redis_connected: bool = False
