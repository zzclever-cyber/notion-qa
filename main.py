"""
企业级 RAG Agent 系统 — FastAPI 应用入口

启动: uvicorn main:app --host 0.0.0.0 --port 8000
Docker: docker compose up -d
"""
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import settings
from core.fsm import AgentFSM, AgentState, AgentContext
from core.intent import IntentClassifier, SlotFiller, IntentType
from core.reflection import BoundedReflection
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.reranker import Reranker
from generation.llm import LLMGenerator
from evaluation.ragas_eval import RagasEvaluator
from session.redis_store import RedisSessionStore
from middleware.tracing import TracingMiddleware, set_session_id
from middleware.metrics import MetricsMiddleware, get_metrics, create_metrics_router
from middleware.rate_limiter import RateLimitMiddleware
from database.session import init_db, AsyncSessionLocal
from database.models import ChatRecord, EvalRecord
from utils.logger import setup_logger, log

# ============================================================
# 全局 Agent 实例
# ============================================================
_agent_instance: Optional["RAGAgent"] = None


def get_agent_instance() -> "RAGAgent":
    """获取全局 RAG Agent 单例"""
    global _agent_instance
    if _agent_instance is None:
        raise RuntimeError("Agent 尚未初始化，请先调用 lifespan 启动")
    return _agent_instance


# ============================================================
# RAG Agent 核心编排器
# ============================================================

class RAGAgent:
    """RAG Agent 核心编排器：协调FSM、检索、生成、自省、评估"""

    def __init__(self):
        self.fsm = AgentFSM()
        self.intent_classifier = IntentClassifier()
        self.slot_filler = SlotFiller()
        self.hybrid_retriever = HybridRetriever(
            bm25=BM25Retriever(),
            faiss=FAISSRetriever(),
            reranker=Reranker(),
        )
        self.llm = LLMGenerator()
        self.reflection = BoundedReflection(llm=self.llm)
        self.evaluator = RagasEvaluator()
        self.session_store = RedisSessionStore()
        self._fallback_sessions: Dict[str, dict] = {}
        # 多租户：每个知识库一份独立检索器，懒加载后缓存（reranker 跨库共享）
        self._retriever_cache: Dict[str, HybridRetriever] = {}

    # ============================================================
    # 生命周期
    # ============================================================

    async def initialize(self):
        """异步初始化：连接 Redis、加载索引"""
        log.info("正在初始化 RAG Agent...")

        # 连接 Redis
        try:
            await self.session_store.connect()
            log.info("Redis 连接成功")
            get_metrics().redis_connected = 1
        except Exception as e:
            log.warning(f"Redis 连接失败，将使用内存存储: {e}")
            get_metrics().redis_connected = 0

        # 加载索引
        try:
            faiss_path = settings.faiss_index_path
            if (faiss_path / "faiss.index").exists():
                self.hybrid_retriever.faiss.load(faiss_path)
                log.info(f"FAISS 索引已加载，向量数: {self.hybrid_retriever.faiss.index.ntotal}")
            else:
                log.warning("FAISS 索引不存在，请先运行 build_index.py")
        except Exception as e:
            log.warning(f"FAISS 索引加载失败: {e}")

        try:
            if settings.bm25_corpus_path.exists():
                self.hybrid_retriever.bm25.load(settings.bm25_corpus_path)
                log.info(f"BM25 索引已加载，文档数: {len(self.hybrid_retriever.bm25.documents)}")
            else:
                log.warning("BM25 索引不存在，请先运行 build_index.py")
        except Exception as e:
            log.warning(f"BM25 索引加载失败: {e}")

        log.info("RAG Agent 初始化完成")

    async def shutdown(self):
        """优雅关闭"""
        await self.session_store.disconnect()
        log.info("RAG Agent 已关闭")

    # ============================================================
    # 多租户检索器（每个知识库独立索引）
    # ============================================================

    def _get_retriever(self, kb_id: Optional[str]) -> HybridRetriever:
        """
        按 kb_id 选择检索器：
        - kb_id 为空 → 默认全局库（sample_docs / 评估用，即 self.hybrid_retriever）
        - kb_id 有值 → 懒加载该知识库的独立 FAISS+BM25 索引，缓存复用
        """
        if not kb_id:
            return self.hybrid_retriever

        cached = self._retriever_cache.get(kb_id)
        if cached is not None:
            return cached

        from processing.indexer import kb_faiss_dir, kb_bm25_path, kb_index_exists
        if not kb_index_exists(kb_id):
            raise RuntimeError(f"知识库 {kb_id} 尚无可用索引（请先上传文档并等待处理完成）")

        faiss = FAISSRetriever()
        faiss.load(kb_faiss_dir(kb_id))
        bm25 = BM25Retriever()
        bm25.load(kb_bm25_path(kb_id))
        # Reranker 无状态，跨知识库共享同一实例，避免重复占内存
        retriever = HybridRetriever(bm25=bm25, faiss=faiss, reranker=self.hybrid_retriever.reranker)
        self._retriever_cache[kb_id] = retriever
        log.info(f"[Agent] 已加载知识库检索器 kb_id={kb_id}（缓存 {len(self._retriever_cache)} 个）")
        return retriever

    def invalidate_retriever(self, kb_id: str):
        """索引变更（上传/删除文档）后调用，下次检索会重新加载"""
        if self._retriever_cache.pop(kb_id, None) is not None:
            log.info(f"[Agent] 已失效知识库检索器缓存 kb_id={kb_id}")

    @staticmethod
    def _estimate_tokens(*texts: str) -> int:
        """粗略估算 token 消耗（中英文混合，约 2 字符/token）。用于 /usage/stats 展示。"""
        total_chars = sum(len(t or "") for t in texts)
        return total_chars // 2

    @staticmethod
    def _slot_params_dict(intent_result) -> dict:
        """把 intent_result.slot_params 归一化成 dict（可能是 dict 或 dataclass）"""
        if not intent_result or not hasattr(intent_result, "slot_params"):
            return {}
        sp = intent_result.slot_params
        if isinstance(sp, dict):
            return sp
        return getattr(sp, "__dict__", {})

    # ============================================================
    # 核心流程
    # ============================================================

    async def run(
        self,
        session_id: str,
        query: str,
        enable_reflection: bool = True,
        kb_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的 RAG 流程 (FSM驱动)
        """
        metrics = get_metrics()
        ctx = self.fsm.start(session_id, query)
        t_total = time.time()

        # 设置日志上下文
        set_session_id(session_id)

        # ── State 1: IDLE → INTENT ──
        self.fsm.transition(AgentState.INTENT)

        t0 = time.time()
        intent_result = self.intent_classifier.classify(query)
        slot_params = self.slot_filler.fill(query, intent_result.intent)
        ctx.intent_result = intent_result
        ctx.timings["intent_ms"] = int((time.time() - t0) * 1000)

        metrics.record_answer(intent_result.intent.value)

        # 闲聊快速通道
        if intent_result.intent == IntentType.CHITCHAT:
            self.fsm.transition(AgentState.DONE)
            ctx.generated_answer = "您好！我是企业知识库智能助手，可以帮您查询公司政策、制度规定、产品信息等。请问有什么可以帮您的？"
            return self._build_response(ctx)

        # ── State 2: INTENT → RETRIEVE ──
        self.fsm.transition(AgentState.RETRIEVE)

        retriever = self._get_retriever(kb_id)
        t0 = time.time()
        retrieved = await retriever.retrieve(
            query=query,
            bm25_top_k=settings.bm25_top_k,
            faiss_top_k=settings.faiss_top_k,
            merge_top_k=settings.merge_top_k,
            rerank_top_k=settings.rerank_top_k,
        )
        ctx.retrieved_docs = retrieved
        retrieve_ms = int((time.time() - t0) * 1000)
        ctx.timings["retrieve_ms"] = retrieve_ms
        metrics.record_retrieval(retrieve_ms)

        # 构建上下文
        context_str = retriever.get_context_for_llm(retrieved)

        # ── 相关性门槛：最高分不达标则直接拒答，不经过 LLM ──
        top_score = max((d.faiss_score for d in retrieved), default=0.0)
        if top_score < settings.min_relevance_score:
            self.fsm.transition(AgentState.DONE)
            ctx.generated_answer = (
                "根据现有知识库信息，无法回答该问题。"
                "建议您查阅公司内部文档或联系相关部门获取更详细的信息。"
            )
            ctx.timings["total_ms"] = int((time.time() - t_total) * 1000)
            await self._persist_session(
                ctx, user_id=user_id, kb_id=kb_id,
                total_tokens=self._estimate_tokens(query, ctx.generated_answer, context_str),
            )
            return self._build_response(ctx)

        # ── State 3: RETRIEVE → REASON ──
        self.fsm.transition(AgentState.REASON)

        t0 = time.time()
        initial_answer = await self.llm.generate_async(
            query=query,
            context=context_str,
            intent=intent_result.intent.value,
            slot_params=str(slot_params.__dict__),
        )
        ctx.generated_answer = initial_answer
        generate_ms = int((time.time() - t0) * 1000)
        ctx.timings["generate_ms"] = generate_ms
        metrics.record_generation(generate_ms)

        # ── State 4: REASON → VERIFY ──
        if enable_reflection:
            self.fsm.transition(AgentState.VERIFY)

            t0 = time.time()
            reflection_result = await self.reflection.reflect_async(
                query=query,
                context=context_str,
                initial_answer=initial_answer,
                intent=intent_result.intent.value,
            )
            ctx.generated_answer = reflection_result.final_answer
            ctx.reflection_rounds = reflection_result.rounds
            ctx.reflection_notes = reflection_result.history
            ctx.conflict_flags = reflection_result.conflict_markers
            ctx.timings["reflection_ms"] = int((time.time() - t0) * 1000)

            metrics.record_reflection(reflection_result.rounds)

        # ── State 5: VERIFY → DONE ──
        self.fsm.transition(AgentState.DONE)
        ctx.timings["total_ms"] = int((time.time() - t_total) * 1000)

        # ── 评估 ──
        self._record_eval(ctx, context_str)

        # ── 持久化 ──
        await self._persist_session(
            ctx, user_id=user_id, kb_id=kb_id,
            total_tokens=self._estimate_tokens(query, ctx.generated_answer, context_str),
        )

        return self._build_response(ctx)

    # ============================================================
    # 会话管理
    # ============================================================

    async def get_session(self, session_id: str) -> Optional[dict]:
        if self.session_store.is_connected:
            return await self.session_store.get_session(session_id)
        return self._fallback_sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        if self.session_store.is_connected:
            return await self.session_store.delete_session(session_id)
        return self._fallback_sessions.pop(session_id, None) is not None

    async def list_active_sessions(self) -> List[str]:
        if self.session_store.is_connected:
            return await self.session_store.get_active_sessions()
        return list(self._fallback_sessions.keys())

    async def _persist_session(
        self,
        ctx: AgentContext,
        user_id: Optional[str] = None,
        kb_id: Optional[str] = None,
        total_tokens: int = 0,
    ):
        """持久化会话数据 — Redis（热存储）+ SQL（持久化）双写"""
        data = {
            "session_id": ctx.session_id,
            "state": self.fsm.state.value,
            "query": ctx.query,
            "intent": ctx.intent_result.intent.value if ctx.intent_result else "",
            "answer": ctx.generated_answer,
            "reflection_rounds": str(ctx.reflection_rounds),
            "trace": ctx.trace,
            "timings": ctx.timings,
        }

        # Redis 热存储（会话查询用）
        try:
            if self.session_store.is_connected:
                await self.session_store.update_session(ctx.session_id, data)
            else:
                self._fallback_sessions[ctx.session_id] = data
        except Exception as e:
            log.warning(f"Redis 会话保存失败，降级到内存存储: {e}")
            self._fallback_sessions[ctx.session_id] = data

        # SQL 持久化（对话历史 & 评估分析用）
        try:
            async with AsyncSessionLocal() as db_session:
                record = ChatRecord(
                    session_id=ctx.session_id,
                    user_id=user_id,
                    kb_id=kb_id,
                    query=ctx.query,
                    answer=ctx.generated_answer,
                    intent=ctx.intent_result.intent.value if ctx.intent_result else "",
                    slot_params=self._slot_params_dict(ctx.intent_result),
                    documents_used=[item.doc_id for item in ctx.retrieved_docs],
                    reflection_rounds=ctx.reflection_rounds,
                    reflection_notes=ctx.reflection_notes,
                    conflict_warning=len(ctx.conflict_flags) > 0,
                    conflict_markers=ctx.conflict_flags,
                    intent_ms=ctx.timings.get("intent_ms", 0),
                    retrieve_ms=ctx.timings.get("retrieve_ms", 0),
                    generate_ms=ctx.timings.get("generate_ms", 0),
                    reflection_ms=ctx.timings.get("reflection_ms", 0),
                    total_ms=ctx.timings.get("total_ms", 0),
                    total_tokens=total_tokens,
                    trace=ctx.trace,
                )
                db_session.add(record)
                await db_session.commit()
        except Exception as e:
            log.warning(f"SQL 持久化失败（Redis 数据不受影响）: {type(e).__name__}: {e}")

    # ============================================================
    # 评估
    # ============================================================

    async def run_stream(
        self,
        session_id: str,
        query: str,
        enable_reflection: bool = True,
        kb_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        SSE 流式版本的 RAG 流程

        每个阶段完成后 yield 一个 SSE 事件，
        在 REASON 阶段逐 token 流式输出生成内容。
        """
        import json as json_mod

        ctx = self.fsm.start(session_id, query)
        t_total = time.time()
        set_session_id(session_id)

        # ── INTENT ──
        self.fsm.transition(AgentState.INTENT)
        intent_result = self.intent_classifier.classify(query)
        slot_params = self.slot_filler.fill(query, intent_result.intent)
        ctx.intent_result = intent_result
        ctx.timings["intent_ms"] = int((time.time() - t_total) * 1000)

        yield f"event: intent\ndata: {json_mod.dumps({'intent': intent_result.intent.value, 'confidence': intent_result.confidence}, ensure_ascii=False)}\n\n"

        if intent_result.intent == IntentType.CHITCHAT:
            self.fsm.transition(AgentState.DONE)
            yield f"event: done\ndata: {json_mod.dumps({'answer': '您好！我是企业知识库智能助手，有什么可以帮您的？'}, ensure_ascii=False)}\n\n"
            return

        # ── RETRIEVE ──
        self.fsm.transition(AgentState.RETRIEVE)
        retriever = self._get_retriever(kb_id)
        t0 = time.time()
        retrieved = await retriever.retrieve(
            query=query,
            bm25_top_k=settings.bm25_top_k,
            faiss_top_k=settings.faiss_top_k,
            merge_top_k=settings.merge_top_k,
            rerank_top_k=settings.rerank_top_k,
        )
        ctx.retrieved_docs = retrieved
        ctx.timings["retrieve_ms"] = int((time.time() - t0) * 1000)
        context_str = retriever.get_context_for_llm(retrieved)

        doc_ids = [item.doc_id for item in retrieved]
        sources = [
            {
                "doc_id": item.doc_id,
                "title": item.doc.get("title", ""),
                "category": item.doc.get("category", ""),
                "snippet": (item.doc.get("content", "") or "")[:200],
                "score": round(float(item.final_score or item.rrf_score or 0.0), 3),
            }
            for item in retrieved[:5]
        ]
        yield f"event: retrieve\ndata: {json_mod.dumps({'documents_found': len(retrieved), 'doc_ids': doc_ids, 'sources': sources}, ensure_ascii=False)}\n\n"

        # ── REASON (流式生成) ──
        self.fsm.transition(AgentState.REASON)
        yield "event: generate_start\ndata: {}\n\n"

        full_answer = ""
        try:
            for token in self.llm.generate_stream(
                query=query,
                context=context_str,
                intent=intent_result.intent.value,
                slot_params=str(slot_params.__dict__),
            ):
                full_answer += token
                yield f"data: {json_mod.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.error(f"流式生成失败: {e}")
            yield f"event: error\ndata: {json_mod.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return

        ctx.generated_answer = full_answer
        ctx.timings["generate_ms"] = int((time.time() - t0) * 1000)

        # ── VERIFY (自省) ──
        if enable_reflection:
            self.fsm.transition(AgentState.VERIFY)
            yield f"event: verify_start\ndata: {json_mod.dumps({'message': '开始事实核查...'}, ensure_ascii=False)}\n\n"

            t0 = time.time()
            reflection_result = await self.reflection.reflect_async(
                query=query,
                context=context_str,
                initial_answer=full_answer,
                intent=intent_result.intent.value,
            )
            ctx.generated_answer = reflection_result.final_answer
            ctx.reflection_rounds = reflection_result.rounds
            ctx.conflict_flags = reflection_result.conflict_markers
            ctx.timings["reflection_ms"] = int((time.time() - t0) * 1000)

            yield f"event: verify\ndata: {json_mod.dumps({'consistent': reflection_result.is_consistent, 'rounds': reflection_result.rounds, 'contradictions': reflection_result.contradiction_count}, ensure_ascii=False)}\n\n"

        # ── DONE ──
        self.fsm.transition(AgentState.DONE)
        ctx.timings["total_ms"] = int((time.time() - t_total) * 1000)
        await self._persist_session(
            ctx, user_id=user_id, kb_id=kb_id,
            total_tokens=self._estimate_tokens(query, ctx.generated_answer, context_str),
        )

        yield f"event: done\ndata: {json_mod.dumps({'session_id': session_id, 'intent': intent_result.intent.value, 'conflict_warning': len(ctx.conflict_flags) > 0, 'total_ms': ctx.timings['total_ms']}, ensure_ascii=False)}\n\n"

    # ============================================================
    # 评估（保留原有逻辑）
    # ============================================================

    async def evaluate_batch(self, samples) -> list:
        """批量评估"""
        results = []
        for sample in samples:
            try:
                result = await self.run(
                    session_id=f"eval_{sample.id}",
                    query=sample.question,
                    enable_reflection=True,
                )
                self.evaluator.evaluate_single(
                    session_id=f"eval_{sample.id}",
                    query=sample.question,
                    generated_answer=result["answer"],
                    expected_answer=sample.expected_answer,
                    contexts=[result["answer"]],
                    retrieved_doc_ids=result.get("documents_used", []),
                    relevant_doc_ids=sample.relevant_doc_ids,
                )
            except Exception as e:
                log.error(f"评估样本失败 {sample.id}: {e}")
        return self.evaluator.results

    # ============================================================
    # 内部方法
    # ============================================================

    def _build_response(self, ctx: AgentContext) -> Dict[str, Any]:
        """构建API响应"""
        doc_ids = [item.doc_id for item in ctx.retrieved_docs]
        return {
            "session_id": ctx.session_id,
            "query": ctx.query,
            "answer": ctx.generated_answer,
            "intent": ctx.intent_result.intent.value if ctx.intent_result else "",
            "documents_used": doc_ids,
            "reflection_rounds": ctx.reflection_rounds,
            "conflict_warning": len(ctx.conflict_flags) > 0,
            "reflection_notes": ctx.reflection_notes,
            "trace": ctx.trace,
            "timings": ctx.timings,
        }

    def _record_eval(self, ctx: AgentContext, context_str: str):
        """记录评估指标"""
        try:
            self.evaluator.evaluate_single(
                session_id=ctx.session_id,
                query=ctx.query,
                generated_answer=ctx.generated_answer,
                expected_answer="",
                contexts=[context_str],
                retrieved_doc_ids=[item.doc_id for item in ctx.retrieved_docs],
                relevant_doc_ids=[],
            )
        except Exception as e:
            log.debug(f"评估记录失败: {e}")


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    setup_logger()
    log.info("正在启动 RAG Agent 服务...")

    # 初始化数据库
    try:
        await init_db()
        log.info("数据库已就绪")
    except Exception as e:
        log.warning(f"数据库初始化失败（服务仍可运行）: {e}")

    global _agent_instance
    _agent_instance = RAGAgent()
    await _agent_instance.initialize()

    log.info("=" * 50)
    log.info("RAG Agent 服务已就绪")
    log.info("  API 文档: http://localhost:8000/docs")
    log.info("  健康检查: http://localhost:8000/health")
    log.info("  监控指标: http://localhost:8000/metrics")
    log.info("=" * 50)

    yield

    if _agent_instance:
        await _agent_instance.shutdown()


app = FastAPI(
    title="企业级 RAG Agent 知识库系统",
    description="基于多路召回与自省机制的企业级 RAG Agent — 完整覆盖检索→推理→评估闭环",
    version="1.0.0",
    lifespan=lifespan,
)

# ── 中间件注册（后添加的先执行） ──

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求追踪（Request ID 注入与全链路追踪）
app.add_middleware(TracingMiddleware)

# 指标采集（HTTP 请求计数与耗时）
app.add_middleware(MetricsMiddleware)

# 限流（IP 级别滑动窗口）
app.add_middleware(RateLimitMiddleware)

# ── 全局异常处理器 ──

from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局未捕获异常处理 — 统一错误响应格式"""
    from middleware.tracing import get_request_id

    log.bind(request_id=get_request_id()).error(
        f"未处理的异常 [{request.method} {request.url.path}]: {exc}",
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "服务器内部错误，请稍后重试",
            "request_id": get_request_id(),
        },
    )


# ── 注册路由 ──

# 鉴权路由（注册/登录/OAuth 桥接）
from api.auth_routes import router as auth_router
app.include_router(auth_router)

# 知识库业务路由（KB CRUD / 上传 / 文档 / 用量）
from api.kb_routes import router as kb_router
app.include_router(kb_router)

# 核心 API 路由
from api.routes import router as api_router
app.include_router(api_router)

# 可观测性路由（Prometheus 指标）
metrics_router = create_metrics_router()
app.include_router(metrics_router)

# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
        reload=True,
    )
