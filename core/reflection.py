"""
受限自省 (Bounded Self-Reflection) 机制

流程：
1. 生成首轮答案
2. 自动触发一次事实核查（Fact-Check）子流程
3. 对答案中的每个声明与检索到的上下文逐条对齐
4. 若发现矛盾，启动最多 2 轮纠错重试
5. 最终输出中明确标注"部分信息存在冲突，请核实"

设计要点：
- 硬上限 2 轮重试，避免无限自纠死循环
- 每轮都有明确的 fact-check → correct 闭环
- 矛盾无法解决时，标注冲突而非强行回答
"""
from typing import Optional
from dataclasses import dataclass, field
from config.settings import settings
from generation.llm import LLMGenerator
from utils.logger import log


@dataclass
class ReflectionResult:
    """自省结果"""
    final_answer: str
    rounds: int = 0
    is_consistent: bool = True
    contradiction_count: int = 0
    conflict_markers: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


class BoundedReflection:
    """
    受限自省机制
    最多 2 轮纠错，避免无限自纠死循环
    """

    MAX_ROUNDS: int = 2

    def __init__(self, llm: Optional[LLMGenerator] = None):
        self.llm = llm or LLMGenerator()
        self.max_rounds = settings.max_reflection_rounds

    def reflect(
        self,
        query: str,
        context: str,
        initial_answer: str,
        intent: str = "factual",
    ) -> ReflectionResult:
        """
        执行受限自省流程
        Args:
            query: 用户原始问题
            context: 检索到的上下文
            initial_answer: 首轮生成的答案
            intent: 意图类型
        Returns:
            ReflectionResult
        """
        result = ReflectionResult(
            final_answer=initial_answer,
            rounds=0,
        )

        current_answer = initial_answer
        all_issues: list[str] = []

        for round_num in range(1, self.max_rounds + 1):
            log.info(f"[Reflection] 开始第 {round_num} 轮事实核查")

            # Step 1: 事实核查
            fact_check = self.llm.fact_check(
                query=query,
                answer=current_answer,
                context=context,
            )

            verdict = fact_check.get("verdict", "partial")
            contradiction_count = fact_check.get("contradiction_count", 0)
            claims = fact_check.get("claims", [])

            result.history.append({
                "round": round_num,
                "verdict": verdict,
                "contradiction_count": contradiction_count,
                "claims": claims,
            })

            log.info(
                f"[Reflection] 第{round_num}轮核查结果: "
                f"verdict={verdict}, contradictions={contradiction_count}"
            )

            # Step 2: 判断是否需要纠正
            if verdict == "consistent" and contradiction_count == 0:
                # 完全一致，无需纠正
                result.is_consistent = True
                result.final_answer = current_answer
                result.rounds = round_num
                log.info("[Reflection] 事实核查通过，答案完全一致")
                break

            # Step 3: 收集问题声明
            contradictions = [
                c for c in claims
                if c.get("status") == "contradicted"
            ]
            not_found = [
                c for c in claims
                if c.get("status") == "not_found"
            ]

            # 无实质矛盾（如 LLM 返回空内容导致 fallback）则跳过纠正
            if not contradictions and not not_found:
                result.is_consistent = True
                result.final_answer = current_answer
                result.rounds = round_num
                log.info("[Reflection] 未发现实质矛盾，跳过纠正")
                break

            issues_text = self._format_issues(contradictions, not_found)
            all_issues.append(issues_text)

            # Step 4: 如果是最后一轮，不再纠错，直接标注冲突
            if round_num == self.max_rounds:
                log.warning(
                    f"[Reflection] 已达最大轮次({self.max_rounds})，"
                    f"仍有{contradiction_count}处矛盾，标注冲突"
                )
                conflict_marker = (
                    "⚠️ [部分信息存在冲突，请核实] 以下声明与知识库信息不一致：\n"
                    + issues_text
                )
                result.final_answer = (
                    current_answer + "\n\n---\n" + conflict_marker
                )
                result.is_consistent = False
                result.contradiction_count = contradiction_count
                result.conflict_markers.append(conflict_marker)
                result.rounds = round_num
                break

            # Step 5: 纠正重生成
            log.info(f"[Reflection] 第{round_num}轮发现矛盾，启动纠正重生成")
            corrected_answer = self.llm.correct(
                query=query,
                context=context,
                issues=issues_text,
            )
            current_answer = corrected_answer

        result.rounds = result.rounds or self.max_rounds
        result.contradiction_count = result.history[-1].get("contradiction_count", 0) if result.history else 0

        return result

    def _format_issues(
        self,
        contradictions: list[dict],
        not_found: list[dict],
    ) -> str:
        """格式化问题声明为可读文本"""
        lines = []

        if contradictions:
            lines.append("## 与上下文矛盾的信息：")
            for i, c in enumerate(contradictions, 1):
                stmt = c.get("statement", "")
                correction = c.get("correction", "无修正建议")
                lines.append(f'{i}. 回答声称: "{stmt}"')
                lines.append(f"   修正建议: {correction}")

        if not_found:
            lines.append("## 上下文中无法验证的信息：")
            for i, c in enumerate(not_found, 1):
                stmt = c.get("statement", "")
                lines.append(f'{i}. "{stmt}" — 此声明在上下文中查无实据')

        return "\n".join(lines)

    async def reflect_async(
        self,
        query: str,
        context: str,
        initial_answer: str,
        intent: str = "factual",
    ) -> ReflectionResult:
        """异步版本（使用线程池执行同步LLM调用）"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.reflect,
            query,
            context,
            initial_answer,
            intent,
        )
