"""
有限状态机 (FSM) 引擎
驱动 Agent 推理流程，严格定义状态转移：
IDLE → INTENT → RETRIEVE → REASON → VERIFY → DONE

两阶段意图设计：IntentClassifier (枚举) + SlotFiller (参数提取) 解耦
新增意图仅需扩展分类器标签，无需修改状态转移表
"""
from enum import Enum
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
import time
from utils.logger import log


class AgentState(str, Enum):
    """Agent 状态枚举"""
    IDLE = "idle"               # 空闲，等待输入
    INTENT = "intent"           # 意图识别阶段
    RETRIEVE = "retrieve"       # 知识检索阶段
    REASON = "reason"           # 逻辑推理与生成阶段
    VERIFY = "verify"           # 事实核查与自省阶段
    DONE = "done"               # 完成，输出结果
    ERROR = "error"             # 异常终止


# 状态转移表 — 定义合法的状态转移
# 格式: { 当前状态: [允许的下一个状态列表] }
STATE_TRANSITIONS: Dict[AgentState, list[AgentState]] = {
    AgentState.IDLE: [AgentState.INTENT, AgentState.ERROR],
    AgentState.INTENT: [AgentState.RETRIEVE, AgentState.DONE, AgentState.ERROR],
    AgentState.RETRIEVE: [AgentState.REASON, AgentState.ERROR],
    AgentState.REASON: [AgentState.VERIFY, AgentState.DONE, AgentState.ERROR],
    AgentState.VERIFY: [AgentState.REASON, AgentState.DONE, AgentState.ERROR],
    AgentState.DONE: [AgentState.IDLE],
    AgentState.ERROR: [AgentState.IDLE],
}


@dataclass
class AgentContext:
    """Agent 上下文 — 贯穿整个推理流程的数据载体"""
    session_id: str = ""
    query: str = ""
    intent_result: Optional[Any] = None   # IntentResult
    retrieved_docs: list = field(default_factory=list)
    generated_answer: str = ""
    reflection_rounds: int = 0
    reflection_notes: list = field(default_factory=list)
    conflict_flags: list = field(default_factory=list)
    eval_metrics: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    error_message: str = ""
    timings: dict = field(default_factory=dict)


class AgentFSM:
    """
    Agent 有限状态机
    管理推理流程的状态转移，每一步推理可控、可追溯、可中断
    """

    def __init__(self):
        self.state: AgentState = AgentState.IDLE
        self.context: Optional[AgentContext] = None
        self._hooks: Dict[AgentState, list[Callable]] = {
            state: [] for state in AgentState
        }
        self._state_start_time: float = 0.0

    def start(self, session_id: str, query: str) -> AgentContext:
        """启动新的推理流程"""
        self.context = AgentContext(session_id=session_id, query=query)
        self.state = AgentState.IDLE
        self._state_start_time = time.time()
        log.info(f"[FSM] 启动新会话: {session_id}, 查询: {query[:50]}...")
        return self.context

    def transition(self, next_state: AgentState) -> bool:
        """
        执行状态转移
        Args:
            next_state: 目标状态
        Returns:
            True = 转移成功, False = 非法转移
        """
        allowed = STATE_TRANSITIONS.get(self.state, [])
        if next_state not in allowed:
            log.error(
                f"[FSM] 非法状态转移: {self.state.value} → {next_state.value}. "
                f"允许: {[s.value for s in allowed]}"
            )
            return False

        elapsed = time.time() - self._state_start_time
        old_state = self.state
        self.state = next_state
        self._state_start_time = time.time()

        # 记录trace
        if self.context:
            self.context.trace.append({
                "from": old_state.value,
                "to": next_state.value,
                "elapsed_ms": int(elapsed * 1000),
            })

        log.debug(f"[FSM] 状态转移: {old_state.value} → {next_state.value} ({elapsed:.3f}s)")
        return True

    def on_enter(self, state: AgentState, callback: Callable):
        """注册状态进入钩子"""
        self._hooks[state].append(callback)

    async def run_hooks(self, state: AgentState):
        """执行指定状态的进入钩子"""
        import asyncio
        for hook in self._hooks[state]:
            if asyncio.iscoroutinefunction(hook):
                await hook(self.context)
            else:
                hook(self.context)

    def get_trace(self) -> list[dict]:
        """获取完整推理追踪"""
        return self.context.trace if self.context else []

    def reset(self):
        """重置状态机"""
        self.state = AgentState.IDLE
        self.context = None
        self._state_start_time = time.time()

    @property
    def is_terminal(self) -> bool:
        """是否处于终止状态"""
        return self.state in (AgentState.DONE, AgentState.ERROR)

    @property
    def can_retrieve(self) -> bool:
        """是否可以进行检索"""
        return self.state in (AgentState.INTENT,)

    @property
    def can_reason(self) -> bool:
        """是否可以进行推理"""
        return self.state in (AgentState.RETRIEVE, AgentState.VERIFY)
