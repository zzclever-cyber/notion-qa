"""
FastAPI 路由定义 — API v1
企业级 RAG Agent 核心接口
"""
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from api.schemas import (
    ChatRequest, ChatResponse,
    SessionInfo, SessionListResponse,
    EvalRequest, EvalResponse,
    HealthResponse,
    PaginatedResponse,
    FeedbackRequest, FeedbackResponse,
)
from config.settings import settings
from middleware.tracing import set_session_id
from middleware.auth import get_optional_user_id
from utils.logger import log

router = APIRouter(prefix="/api/v1", tags=["RAG Agent v1"])


# 延迟导入，避免循环依赖
def get_agent():
    from main import get_agent_instance
    return get_agent_instance()


# ============================================================
# 聊天端点
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: Optional[str] = Depends(get_optional_user_id)):
    """
    主问答接口

    完整流程：意图识别 → 知识检索 → 逻辑推理 → 自省校验 → 输出

    - **query**: 用户问题（必填，1-2000字符）
    - **session_id**: 会话ID（可选，不传则自动创建）
    - **kb_id**: 知识库ID（可选，指定则只在该库检索；不传用默认全局库）
    - **enable_reflection**: 是否启用自省校验（默认开启）

    鉴权可选：带 Bearer token 则关联用户与其知识库，匿名请求走默认库。
    """
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    set_session_id(session_id)

    try:
        agent = get_agent()
        result = await agent.run(
            session_id=session_id,
            query=request.query,
            enable_reflection=request.enable_reflection,
            kb_id=request.kb_id,
            user_id=user_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        log.error(f"聊天请求失败 [{session_id}]: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    kb_id: Optional[str] = Query(None, description="知识库ID，覆盖请求体中的 kb_id"),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """
    流式问答接口（SSE — Server-Sent Events）

    真实事件流：intent → retrieve → generate(token级) → verify → done
    每个阶段以 SSE named event 推送，生成阶段逐 token 流式输出

    kb_id 可从查询参数或请求体传入（查询参数优先，便于前端 useChat 拼 URL）。
    """
    effective_kb_id = kb_id or request.kb_id

    async def event_stream():
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        set_session_id(session_id)
        try:
            agent = get_agent()
            async for event in agent.run_stream(
                session_id=session_id,
                query=request.query,
                enable_reflection=request.enable_reflection,
                kb_id=effective_kb_id,
                user_id=user_id,
            ):
                yield event
        except Exception as e:
            import json
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# 反馈闭环 — 用户评价回答质量
# ============================================================

@router.post("/chat/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    提交用户反馈 — 形成业务闭环

    反馈数据持久化到数据库，用于：
    - 监控答案质量趋势
    - 定位检索/生成环节问题
    - 驱动评测集更新与 Prompt 优化
    """
    from database.session import AsyncSessionLocal
    from database.models import ChatRecord
    from sqlalchemy import update

    try:
        async with AsyncSessionLocal() as db_session:
            # 更新对应会话记录的反馈字段
            stmt = (
                update(ChatRecord)
                .where(ChatRecord.session_id == feedback.session_id)
                .values(
                    metadata_=ChatRecord.metadata_.op("||")(
                        {
                            "user_feedback": feedback.rating,
                            "feedback_comment": feedback.comment or "",
                            "feedback_tags": feedback.tags or [],
                        }
                    ),
                )
            )
            await db_session.execute(stmt)
            await db_session.commit()
    except Exception as e:
        log.warning(f"反馈持久化失败（不影响用户体验）: {e}")

    log.info(
        f"[反馈] session={feedback.session_id} rating={feedback.rating}"
        + (f" tags={feedback.tags}" if feedback.tags else "")
        + (f" comment={feedback.comment[:50]}" if feedback.comment else "")
    )

    return FeedbackResponse(
        session_id=feedback.session_id,
        status="recorded",
        message="感谢您的反馈，这将帮助我们持续改进回答质量。",
    )


# ============================================================
# 会话管理
# ============================================================

@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """获取指定会话的完整信息"""
    agent = get_agent()
    data = await agent.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return SessionInfo(**data)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    agent = get_agent()
    deleted = await agent.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在或删除失败")
    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=50, ge=1, le=200, description="每页数量"),
):
    """
    列出活跃会话（分页）

    - **page**: 页码（从 1 开始）
    - **page_size**: 每页数量（1-200）
    """
    agent = get_agent()
    all_sessions = await agent.list_active_sessions()

    # 分页
    total = len(all_sessions)
    start = (page - 1) * page_size
    end = start + page_size
    paged = all_sessions[start:end]

    return SessionListResponse(
        active_sessions=paged,
        count=len(paged),
        total=total,
        page=page,
        page_size=page_size,
    )


# ============================================================
# 评估
# ============================================================

@router.post("/eval/run", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest):
    """
    运行基准测试评估

    在 150 条评测集上测量 RAG 三元组指标
    可按 query_type 筛选（single_hop / multi_hop / numerical / negation）
    """
    try:
        agent = get_agent()

        from evaluation.dataset import EvalDataset
        dataset = EvalDataset()
        dataset.load(settings.eval_dataset_path)

        samples = dataset.samples
        if request.query_types:
            samples = [s for s in samples if s.query_type in request.query_types]
        if request.max_samples:
            samples = samples[:request.max_samples]

        log.info(f"[API] 开始评估: {len(samples)} 个样本")

        results = await agent.evaluate_batch(samples)

        n = len(results)
        resp = EvalResponse(
            total_samples=n,
            avg_faithfulness=sum(r.metrics.faithfulness for r in results) / n if n else 0,
            avg_context_precision=sum(r.metrics.context_precision for r in results) / n if n else 0,
            avg_answer_relevance=sum(r.metrics.answer_relevance for r in results) / n if n else 0,
            avg_recall_at_5=sum(r.metrics.recall_at_5 for r in results) / n if n else 0,
        )

        # 按类型统计
        for qtype in ["single_hop", "multi_hop", "numerical", "negation"]:
            typed = [r for r, s in zip(results, samples) if s.query_type == qtype]
            if typed:
                resp.by_query_type[qtype] = {
                    "count": len(typed),
                    "avg_faithfulness": round(sum(r.metrics.faithfulness for r in typed) / len(typed), 4),
                    "avg_recall_at_5": round(sum(r.metrics.recall_at_5 for r in typed) / len(typed), 4),
                }

        return resp

    except Exception as e:
        log.error(f"评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/results")
async def get_eval_results():
    """获取最近一次评估的详细结果"""
    from pathlib import Path
    import json

    results_path = settings.data_dir / "eval_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="评估结果不存在，请先运行 POST /api/v1/eval/run")

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 健康检查
# ============================================================

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    健康检查端点

    返回各组件状态：
    - retrievers: FAISS / BM25 是否就绪
    - redis: Redis 连接状态
    """
    agent = get_agent()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        retrievers={
            "bm25": agent.hybrid_retriever.bm25.is_ready,
            "faiss": agent.hybrid_retriever.faiss.is_ready,
        },
        redis_connected=agent.session_store.is_connected if agent.session_store else False,
    )
