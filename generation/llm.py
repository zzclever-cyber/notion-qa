"""
LLM 生成模块
封装 OpenAI 兼容接口的 LLM 调用，包含多阶段 Prompt 模板

关键设计：
- 所有 LLM 调用经过熔断器(CircuitBreaker) + 指数退避重试保护
- generate / fact_check / correct 三个核心方法均接入 resilience 模块
- generate_stream 支持真正的 token-level SSE 流式输出
"""
import json
import time
from typing import Optional, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings
from resilience.circuit_breaker import llm_circuit_breaker, CircuitBreakerOpenError
from resilience.retry import retry_with_backoff
from middleware.metrics import get_metrics
from utils.logger import log


# ============================================================
# Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个企业知识库智能助手，名为"企业RAG Agent"。你的职责是基于提供的上下文信息准确回答用户问题。

请严格遵守以下规则：
1. 回答必须基于提供的上下文信息，不得编造不存在的事实
2. 如果上下文中没有相关信息，请明确说明"根据现有知识库信息，无法回答该问题"
3. 引用信息时，请注明来源文档的标题和类别
4. 涉及数值计算时，请给出详细的计算步骤
5. 使用专业但友好的语气，回答简洁准确
6. 使用中文回答"""

REASON_PROMPT = """请根据以下上下文信息回答用户问题。

上下文信息：
{context}

用户问题：{query}

意图类型：{intent}
相关参数：{slot_params}

请给出准确、完整的回答。涉及数值计算时列明计算步骤，引用信息注明来源。"""

REFLECTION_FACT_CHECK_PROMPT = """你是一个事实核查专家。请逐条检查以下回答中的每个事实声明是否与提供的上下文信息一致。

上下文信息：
{context}

原始回答：
{answer}

请逐条分析：
1. 列出回答中的每个事实声明
2. 对每条声明，判断其是否在上下文中获得支持（supported / contradicted / not_found）
3. 对存在矛盾(contradicted)的声明，指出上下文中的正确信息
4. 对查无实据(not_found)的声明，评估其合理性

请仅输出JSON格式：
{{
  "verdict": "consistent" | "contradicted" | "partial",
  "claims": [
    {{
      "statement": "...",
      "status": "supported" | "contradicted" | "not_found",
      "evidence": "上下文中的相关原文或 null",
      "correction": "修正建议或 null"
    }}
  ],
  "contradiction_count": <number>,
  "summary": "<简短总结>"
}}"""

CORRECTION_PROMPT = """你上一轮的回答存在以下问题：

{issues}

请根据上下文信息重新生成正确的回答：

上下文信息：
{context}

用户问题：{query}

请修正错误，确保所有事实声明都有上下文支撑。对于仍无法确定的信息，请明确标注"部分信息存在冲突，请核实"。"""


# ============================================================
# LLMGenerator — 集成熔断器 + 重试
# ============================================================

class LLMGenerator:
    """
    LLM 生成器

    所有 LLM 调用均经过：
    1. 熔断器(CircuitBreaker) — 连续失败 N 次后熔断，保护下游 API
    2. 指数退避重试 — 网络瞬时故障自动恢复
    3. 指标采集 — 记录调用次数、token 消耗、错误次数
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or settings.llm_model_name
        self.api_base = api_base or settings.llm_api_base
        self.api_key = api_key or settings.llm_api_key
        self.llm = self._create_llm(temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
        self.fast_llm = self._create_llm(temperature=0.1, max_tokens=1024)
        self._metrics = get_metrics()

    def _create_llm(self, temperature: float, max_tokens: int) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ============================================================
    # 核心安全调用原语
    # ============================================================

    def _record_token_usage(self, response) -> int:
        """
        从 LangChain AIMessage 响应中提取真实 token 用量

        兼容新旧版本 LangChain 的两种 metadata 位置：
        - 新版: response.usage_metadata["total_tokens"]
        - 旧版: response.response_metadata["token_usage"]["total_tokens"]
        """
        tokens = 0
        try:
            # 新版 LangChain (>=0.3)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.get("total_tokens", 0)
            # 旧版 LangChain
            elif hasattr(response, 'response_metadata'):
                token_usage = response.response_metadata.get("token_usage", {})
                tokens = token_usage.get("total_tokens", 0) if isinstance(token_usage, dict) else 0
        except Exception:
            pass
        self._metrics.record_llm_call(tokens=tokens)
        return tokens

    def _safe_invoke(self, llm: ChatOpenAI, messages: list) -> str:
        """
        带熔断器 + 重试保护的 LLM 调用

        链路: 熔断器检查 → 重试(3次, 指数退避) → LLM.invoke() → 记录指标
        """
        def _do_invoke():
            return llm.invoke(messages)

        _do_with_retry = retry_with_backoff(max_tries=3, base_delay=0.5, max_delay=5.0)(_do_invoke)

        try:
            response = llm_circuit_breaker.call(_do_with_retry)
            self._record_token_usage(response)
            content = response.content if hasattr(response, 'content') else str(response)
            if content is None:
                result = ""
            elif isinstance(content, str):
                result = content
            elif isinstance(content, list) and len(content) > 0:
                item = content[0]
                if isinstance(item, dict):
                    result = item.get("text") or item.get("content") or str(item)
                else:
                    result = str(item)
            else:
                result = str(content)
            return result.strip()
        except CircuitBreakerOpenError:
            log.error("[LLM] 熔断器已打开，拒绝 LLM 调用")
            self._metrics.record_llm_error()
            raise
        except Exception as e:
            self._metrics.record_llm_error()
            log.error(f"[LLM] 调用失败（重试已耗尽）: {e}")
            raise

    def _safe_invoke_json(self, llm: ChatOpenAI, messages: list) -> dict:
        """
        带保护的 LLM 调用 + JSON 解析
        用于 fact_check 等需要结构化输出的场景
        """
        try:
            raw = self._safe_invoke(llm, messages)
        except Exception as e:
            log.warning(f"[LLM] _safe_invoke 失败: {e}")
            return {
                "verdict": "partial",
                "claims": [],
                "contradiction_count": 0,
                "summary": f"LLM调用失败: {str(e)}",
            }

        # 空响应直接返回兜底结果，防止 json.loads("") 抛异常
        if not raw:
            log.warning("[LLM] LLM 返回空内容，使用兜底结果")
            return {
                "verdict": "partial",
                "claims": [],
                "contradiction_count": 0,
                "summary": "LLM返回空内容，无法完成事实核查",
            }

        # 去除可能的 markdown 代码块标记
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"[LLM] JSON 解析失败: {e}, raw={raw[:300]}")
            return {
                "verdict": "partial",
                "claims": [],
                "contradiction_count": 0,
                "summary": f"JSON解析异常: {str(e)}",
                "raw": raw,
            }

    # ============================================================
    # 生成方法
    # ============================================================

    def generate(
        self,
        query: str,
        context: str,
        intent: str = "factual",
        slot_params: str = "{}",
    ) -> str:
        """
        基于上下文生成回答
        经过熔断器 + 重试保护
        """
        prompt = REASON_PROMPT.format(
            context=context,
            query=query,
            intent=intent,
            slot_params=slot_params,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        log.info(f"[LLM] generate: query={query[:50]}..., intent={intent}")
        t0 = time.time()
        result = self._safe_invoke(self.llm, messages)
        log.info(f"[LLM] generate 完成, 耗时={time.time()-t0:.2f}s, 回答长度={len(result)}")
        return result

    def fact_check(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> dict:
        """
        事实核查：将回答中的每个声明与上下文逐条对齐
        经过熔断器 + 重试保护
        """
        prompt = REFLECTION_FACT_CHECK_PROMPT.format(
            context=context,
            answer=answer,
        )

        messages = [
            SystemMessage(content="你是一个事实核查专家。请只输出JSON。"),
            HumanMessage(content=prompt),
        ]

        log.info(f"[LLM] fact_check: query={query[:50]}..., 回答长度={len(answer)}")
        t0 = time.time()
        result = self._safe_invoke_json(self.fast_llm, messages)
        log.info(
            f"[LLM] fact_check 完成, 耗时={time.time()-t0:.2f}s, "
            f"verdict={result.get('verdict', '?')}, "
            f"contradictions={result.get('contradiction_count', 0)}"
        )
        return result

    def correct(
        self,
        query: str,
        context: str,
        issues: str,
    ) -> str:
        """
        基于核查问题进行纠正重生成
        经过熔断器 + 重试保护
        """
        prompt = CORRECTION_PROMPT.format(
            issues=issues,
            context=context,
            query=query,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        log.info(f"[LLM] correct: query={query[:50]}..., issues_len={len(issues)}")
        t0 = time.time()
        result = self._safe_invoke(self.llm, messages)
        log.info(f"[LLM] correct 完成, 耗时={time.time()-t0:.2f}s")
        return result

    def generate_stream(self, query: str, context: str, intent: str = "factual", slot_params: str = "{}"):
        """
        Token-level SSE 流式生成

        使用 LangChain ChatOpenAI 的 .stream() 方法实现真正的逐 token 输出
        经过熔断器检查（注意：流式模式下重试不适用，由调用方处理）
        """
        prompt = REASON_PROMPT.format(
            context=context,
            query=query,
            intent=intent,
            slot_params=slot_params,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 熔断器检查
        try:
            llm_circuit_breaker._before_request()
        except CircuitBreakerOpenError:
            log.error("[LLM] 熔断器已打开，拒绝流式 LLM 调用")
            self._metrics.record_llm_error()
            yield "data: [ERROR] 服务暂时不可用，请稍后重试\n\n"
            return

        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
            llm_circuit_breaker._on_success()
            self._metrics.record_llm_call()
        except Exception as e:
            llm_circuit_breaker._on_failure()
            self._metrics.record_llm_error()
            log.error(f"[LLM] 流式生成失败: {e}")
            yield "\n\n[生成中断，请重试]"

    # ============================================================
    # Async 包装方法 — 避免阻塞事件循环
    # ============================================================

    async def generate_async(self, query: str, context: str, intent: str = "factual", slot_params: str = "{}") -> str:
        """异步版本：在线程池中执行同步 generate()"""
        import asyncio
        return await asyncio.to_thread(self.generate, query, context, intent, slot_params)

    async def fact_check_async(self, query: str, answer: str, context: str) -> dict:
        """异步版本：在线程池中执行同步 fact_check()"""
        import asyncio
        return await asyncio.to_thread(self.fact_check, query, answer, context)

    async def correct_async(self, query: str, context: str, issues: str) -> str:
        """异步版本：在线程池中执行同步 correct()"""
        import asyncio
        return await asyncio.to_thread(self.correct, query, context, issues)
