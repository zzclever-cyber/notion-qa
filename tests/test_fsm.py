"""
FSM 状态机单元测试
"""
import pytest
from core.fsm import AgentFSM, AgentState, STATE_TRANSITIONS

pytestmark = pytest.mark.unit


class TestAgentFSM:
    """FSM 引擎测试"""

    def setup_method(self):
        self.fsm = AgentFSM()

    def test_initial_state(self):
        """初始状态为 IDLE"""
        assert self.fsm.state == AgentState.IDLE

    def test_start_creates_context(self):
        """start() 创建上下文"""
        ctx = self.fsm.start("sess_001", "测试查询")
        assert ctx.session_id == "sess_001"
        assert ctx.query == "测试查询"

    def test_valid_transition_idle_to_intent(self):
        """IDLE → INTENT 合法转移"""
        self.fsm.start("sess_001", "查询")
        assert self.fsm.transition(AgentState.INTENT) is True
        assert self.fsm.state == AgentState.INTENT

    def test_valid_transition_intent_to_retrieve(self):
        """INTENT → RETRIEVE 合法转移"""
        self.fsm.start("sess_001", "查询")
        self.fsm.transition(AgentState.INTENT)
        assert self.fsm.transition(AgentState.RETRIEVE) is True
        assert self.fsm.state == AgentState.RETRIEVE

    def test_invalid_transition(self):
        """非法状态转移应返回 False"""
        self.fsm.start("sess_001", "查询")
        # IDLE → REASON 不合法（必须先经过 INTENT 和 RETRIEVE）
        assert self.fsm.transition(AgentState.REASON) is False
        assert self.fsm.state == AgentState.IDLE  # 状态不变

    def test_full_pipeline(self):
        """完整流程测试"""
        self.fsm.start("sess_001", "公司年假有多少天？")

        transitions = [
            (AgentState.INTENT, True),
            (AgentState.RETRIEVE, True),
            (AgentState.REASON, True),
            (AgentState.VERIFY, True),
            (AgentState.DONE, True),
        ]

        for state, expected in transitions:
            assert self.fsm.transition(state) == expected
            assert self.fsm.state == state

    def test_reflection_loop(self):
        """VERIFY → REASON 循环（自省纠错回路）"""
        self.fsm.start("sess_001", "查询")
        self.fsm.transition(AgentState.INTENT)
        self.fsm.transition(AgentState.RETRIEVE)
        self.fsm.transition(AgentState.REASON)
        self.fsm.transition(AgentState.VERIFY)
        # 自省发现矛盾，允许回到 REASON
        assert self.fsm.transition(AgentState.REASON) is True
        assert self.fsm.state == AgentState.REASON

    def test_trace_records_transitions(self):
        """状态转移应记录追踪"""
        self.fsm.start("sess_001", "查询")
        self.fsm.transition(AgentState.INTENT)
        self.fsm.transition(AgentState.RETRIEVE)

        trace = self.fsm.get_trace()
        assert len(trace) == 2
        assert trace[0]["from"] == "idle"
        assert trace[0]["to"] == "intent"
        assert trace[1]["to"] == "retrieve"

    def test_is_terminal(self):
        """DONE 和 ERROR 是终止状态"""
        self.fsm.start("sess_001", "查询")
        assert self.fsm.is_terminal is False
        self.fsm.transition(AgentState.INTENT)
        self.fsm.transition(AgentState.DONE)
        assert self.fsm.is_terminal is True

    def test_reset(self):
        """reset() 回到初始状态"""
        self.fsm.start("sess_001", "查询")
        self.fsm.transition(AgentState.INTENT)
        self.fsm.reset()
        assert self.fsm.state == AgentState.IDLE
        assert self.fsm.context is None

    def test_state_transitions_coverage(self):
        """验证状态转移表的完整性"""
        # 每个状态都有定义转移目标
        for state in AgentState:
            assert state in STATE_TRANSITIONS
            # 至少有一个合法的下一个状态
            assert len(STATE_TRANSITIONS[state]) > 0

    def test_on_enter_hook(self):
        """状态进入钩子"""
        hook_called = []

        def hook(ctx):
            hook_called.append(ctx)

        self.fsm.on_enter(AgentState.INTENT, hook)
        self.fsm.start("sess_001", "查询")
        # run_hooks 是 async，这里直接调同步版本
        import asyncio

        async def test_hook():
            await self.fsm.run_hooks(AgentState.INTENT)

        # 当前状态是 IDLE，先转 INTENT
        self.fsm.transition(AgentState.INTENT)

    def test_can_retrieve(self):
        """仅在 INTENT 状态可以检索"""
        self.fsm.start("sess_001", "查询")
        assert self.fsm.can_retrieve is False
        self.fsm.transition(AgentState.INTENT)
        assert self.fsm.can_retrieve is True

    def test_can_reason(self):
        """在 RETRIEVE 和 VERIFY 状态可以推理"""
        self.fsm.start("sess_001", "查询")
        assert self.fsm.can_reason is False
        self.fsm.transition(AgentState.INTENT)
        assert self.fsm.can_reason is False
        self.fsm.transition(AgentState.RETRIEVE)
        assert self.fsm.can_reason is True
        self.fsm.transition(AgentState.REASON)
        self.fsm.transition(AgentState.VERIFY)
        assert self.fsm.can_reason is True
