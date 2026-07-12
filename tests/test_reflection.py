"""
受限自省机制测试
"""
import pytest
from unittest.mock import MagicMock, patch
from core.reflection import BoundedReflection, ReflectionResult

pytestmark = pytest.mark.unit


class TestBoundedReflection:
    """受限自省测试"""

    def setup_method(self):
        self.reflection = BoundedReflection()

    def test_reflection_max_rounds(self):
        """验证最大轮次限制为2"""
        assert self.reflection.max_rounds == 2

    def test_reflection_result_initialization(self):
        """ReflectionResult 初始化"""
        result = ReflectionResult(final_answer="测试答案")
        assert result.final_answer == "测试答案"
        assert result.rounds == 0
        assert result.is_consistent is True
        assert result.contradiction_count == 0

    @patch("core.reflection.LLMGenerator")
    def test_reflection_consistent_answer(self, mock_llm_class):
        """答案一致时无需纠错"""
        mock_llm = mock_llm_class.return_value

        # 模拟事实核查返回一致
        mock_llm.fact_check.return_value = {
            "verdict": "consistent",
            "contradiction_count": 0,
            "claims": [
                {
                    "statement": "核心工作时间为10:00至16:00",
                    "status": "supported",
                    "evidence": "上午10:00至下午16:00",
                    "correction": None,
                }
            ],
            "summary": "所有声明与上下文一致",
        }

        result = self.reflection.reflect(
            query="公司工作时间是什么？",
            context="核心工作时间段为上午10:00至下午16:00。",
            initial_answer="核心工作时间是10:00至16:00。",
        )

        assert result.is_consistent is True
        assert result.rounds == 1  # 一轮核查通过就停止
        mock_llm.fact_check.assert_called_once()
        mock_llm.correct.assert_not_called()  # 一致时无需纠正

    @patch("core.reflection.LLMGenerator")
    def test_reflection_contradicted_answer(self, mock_llm_class):
        """答案有矛盾时触发纠正"""
        mock_llm = mock_llm_class.return_value

        # 第一轮核查：发现矛盾
        mock_llm.fact_check.side_effect = [
            {
                "verdict": "contradicted",
                "contradiction_count": 2,
                "claims": [
                    {
                        "statement": "年假有20天",
                        "status": "contradicted",
                        "evidence": "入职满1年5天，满3年10天",
                        "correction": "入职满1年只有5天",
                    }
                ],
                "summary": "年假天数声明错误",
            },
            # 第二轮核查：修正后一致
            {
                "verdict": "consistent",
                "contradiction_count": 0,
                "claims": [
                    {
                        "statement": "年假有5天",
                        "status": "supported",
                        "evidence": "入职满1年享有5天带薪年假",
                        "correction": None,
                    }
                ],
                "summary": "修正后与上下文一致",
            },
        ]

        # 纠正重生成
        mock_llm.correct.return_value = "修正后：入职满1年的员工年假为5天。"

        result = self.reflection.reflect(
            query="入职1年员工的年假是多少？",
            context="入职满1年享有5天带薪年假。",
            initial_answer="入职满1年的员工年假有20天。",
        )

        assert result.is_consistent is True
        assert result.rounds == 2  # 第一轮发现矛盾，第二轮通过
        assert mock_llm.correct.call_count == 1
        assert mock_llm.fact_check.call_count == 2

    @patch("core.reflection.LLMGenerator")
    def test_reflection_max_rounds_reached(self, mock_llm_class):
        """达到最大轮次后标注冲突"""
        mock_llm = mock_llm_class.return_value

        # 每轮都返回矛盾
        mock_llm.fact_check.return_value = {
            "verdict": "contradicted",
            "contradiction_count": 1,
            "claims": [
                {
                    "statement": "声称X",
                    "status": "contradicted",
                    "evidence": "实际上Y",
                    "correction": "应为Y",
                }
            ],
            "summary": "仍有矛盾",
        }

        result = self.reflection.reflect(
            query="测试问题",
            context="测试上下文",
            initial_answer="有矛盾的答案",
        )

        # 达到最大轮次
        assert result.rounds == 2
        assert result.is_consistent is False
        assert "部分信息存在冲突，请核实" in result.final_answer

    def test_format_issues(self):
        """问题格式化"""
        contradictions = [
            {"statement": "A是B", "status": "contradicted", "correction": "A是C"},
        ]
        not_found = [
            {"statement": "D是E", "status": "not_found"},
        ]

        output = self.reflection._format_issues(contradictions, not_found)
        assert "与上下文矛盾" in output
        assert "A是B" in output
        assert "A是C" in output or "修正" in output
        assert "无法验证" in output
        assert "D是E" in output

    def test_reflection_rounds_capped(self):
        """验证不会再超过 max_rounds + 1 轮"""
        # 通过 patch LLM 确保 fast path
        with patch("core.reflection.LLMGenerator") as mock_llm_class:
            mock_llm = mock_llm_class.return_value
            mock_llm.fact_check.return_value = {
                "verdict": "consistent",
                "contradiction_count": 0,
                "claims": [],
                "summary": "ok",
            }

            result = self.reflection.reflect(
                query="Q", context="C", initial_answer="A"
            )
            # 一致时仅一轮
            assert result.rounds <= 2
            assert result.is_consistent is True
