"""
意图识别模块 - 两阶段设计
Phase 1: 意图分类（IntentClassifier）— 将查询归类到预定义的意图枚举
Phase 2: 参数槽填充（SlotFiller）— 从查询中提取关键参数

两阶段解耦设计：新增意图仅需扩展分类器标签，无需修改FSM状态转移表
"""
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings
from utils.logger import log


class IntentType(str, Enum):
    """意图类型枚举"""
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    NUMERICAL = "numerical"
    NEGATION = "negation"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: IntentType
    confidence: float = 1.0
    slot_params: dict = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class SlotParams:
    """参数槽"""
    entity: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[str] = None
    condition: Optional[str] = None
    time_range: Optional[str] = None
    comparison: Optional[str] = None


# ============================================================
# Prompt 模板
# ============================================================

INTENT_CLASSIFIER_PROMPT = """你是一个查询意图分类器。请将用户查询分类到以下意图类型之一：

意图类型定义：
- factual: 事实查询，答案在单篇文档中可直接找到（如"公司年假有多少天？"）
- multi_hop: 多跳推理，需要结合多篇文档的信息（如"入职3年的员工能参加股权激励吗？年假多少天？"）
- numerical: 数值计算，涉及数字运算或比较（如"3年专业版总费用是多少？"）
- negation: 否定反问（如"年假可以跨年累积吗？"）
- chitchat: 闲聊问候（如"你好""今天天气怎么样"）
- unknown: 无法判断意图

请仅输出JSON格式，不要包含其他内容：
{{
  "intent": "<intent_type>",
  "confidence": <0.0-1.0>,
  "reasoning": "<简短分类理由>"
}}

用户查询：{query}
"""

SLOT_FILLER_PROMPT = """你是一个信息抽取器。请从用户查询中提取关键参数。

查询：{query}
意图类型：{intent}

请仅输出JSON格式：
{{
  "entity": "... or null",
  "attribute": "... or null",
  "value": "... or null",
  "condition": "... or null",
  "time_range": "... or null",
  "comparison": "... or null"
}}
"""


def _parse_llm_json(raw_text: str) -> dict:
    """从 LLM 返回文本中提取 JSON"""
    import json

    text = raw_text.strip()
    # 空内容直接抛异常给上层兜底
    if not text:
        raise ValueError("LLM 返回空内容，无法解析 JSON")

    # 去除 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # 尝试找到 JSON 对象边界
    if "{" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        text = text[start:end]

    return json.loads(text)


# ============================================================
# IntentClassifier — 使用 LLMGenerator（复用熔断+重试）
# ============================================================

class IntentClassifier:
    """意图分类器"""

    def __init__(self, llm_generator=None):
        """
        Args:
            llm_generator: LLMGenerator 实例，为 None 时创建独立 ChatOpenAI
        """
        self._llm_gen = llm_generator

    @property
    def llm_gen(self):
        """延迟获取 LLMGenerator 避免循环导入"""
        if self._llm_gen is None:
            from generation.llm import LLMGenerator
            self._llm_gen = LLMGenerator()
        return self._llm_gen

    def classify(self, query: str) -> IntentResult:
        """意图分类 — 规则优先，LLM兜底"""
        if self._is_chitchat(query):
            return IntentResult(intent=IntentType.CHITCHAT, confidence=0.99, reasoning="规则匹配：闲聊问候")
        if self._is_negation(query):
            return IntentResult(intent=IntentType.NEGATION, confidence=0.85, reasoning="规则匹配：否定/反问句式")

        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)
            messages = [
                SystemMessage(content="你是一个专业的查询意图分类器。请只输出JSON。"),
                HumanMessage(content=prompt),
            ]
            raw = self.llm_gen._safe_invoke(self.llm_gen.fast_llm, messages)
            log.info(f"[Intent] LLM返回: {repr(raw[:300])}")
            result = _parse_llm_json(raw)
            return IntentResult(
                intent=IntentType(result.get("intent", "unknown")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            log.warning(f"LLM意图分类失败: type={type(e).__name__}, {e}，使用规则兜底")
            return self._rule_fallback(query)

    def _is_chitchat(self, query: str) -> bool:
        patterns = ["你好", "谢谢", "再见", "今天天气", "你是谁", "hello", "hi", "thanks", "bye"]
        return any(p in query.lower().strip() for p in patterns)

    def _is_negation(self, query: str) -> bool:
        markers = ["是否", "是不是", "能不能", "可以吗", "有没有", "不会", "不可以", "不允许", "不能", "没有", "吗", "难道"]
        return any(m in query for m in markers)

    def _rule_fallback(self, query: str) -> IntentResult:
        if any(w in query for w in ["多少", "计算", "总共", "折", "优惠后", "差"]):
            return IntentResult(intent=IntentType.NUMERICAL, confidence=0.5, reasoning="规则兜底：含数值关键词")
        if any(w in query for w in ["是否", "吗", "不能", "没有"]):
            return IntentResult(intent=IntentType.NEGATION, confidence=0.5, reasoning="规则兜底：含反问/否定关键词")
        return IntentResult(intent=IntentType.FACTUAL, confidence=0.3, reasoning="规则兜底：默认事实查询")


# ============================================================
# SlotFiller — 同样复用 LLMGenerator
# ============================================================

class SlotFiller:
    """参数槽填充器"""

    def __init__(self, llm_generator=None):
        self._llm_gen = llm_generator

    @property
    def llm_gen(self):
        if self._llm_gen is None:
            from generation.llm import LLMGenerator
            self._llm_gen = LLMGenerator()
        return self._llm_gen

    def fill(self, query: str, intent: IntentType) -> SlotParams:
        try:
            prompt = SLOT_FILLER_PROMPT.format(query=query, intent=intent.value)
            messages = [
                SystemMessage(content="你是一个信息抽取器。请只输出JSON。"),
                HumanMessage(content=prompt),
            ]
            raw = self.llm_gen._safe_invoke(self.llm_gen.fast_llm, messages)
            log.info(f"[SlotFill] LLM返回: {repr(raw[:300])}")
            result = _parse_llm_json(raw)
            return SlotParams(
                entity=result.get("entity"),
                attribute=result.get("attribute"),
                value=result.get("value"),
                condition=result.get("condition"),
                time_range=result.get("time_range"),
                comparison=result.get("comparison"),
            )
        except Exception as e:
            log.warning(f"参数槽填充失败: type={type(e).__name__}, {e}，返回空槽位")
            return SlotParams()
