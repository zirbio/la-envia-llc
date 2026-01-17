# tests/analyzers/test_claude_result.py
import pytest
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel


class TestClaudeAnalysisResult:
    def test_create_result(self):
        result = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            risk_factors=["earnings_in_3_weeks"],
            context_summary="Large call sweep indicates institutional accumulation",
            recommendation="valid_catalyst",
            reasoning="Sweep size and timing suggest informed buying",
        )
        assert result.catalyst_type == CatalystType.INSTITUTIONAL_FLOW
        assert result.risk_level == RiskLevel.LOW
        assert len(result.risk_factors) == 1

    def test_catalyst_types(self):
        for catalyst in CatalystType:
            result = ClaudeAnalysisResult(
                catalyst_type=catalyst,
                catalyst_confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                context_summary="test",
                recommendation="test",
            )
            assert result.catalyst_type == catalyst

    def test_is_actionable(self):
        high_conf_result = ClaudeAnalysisResult(
            catalyst_type=CatalystType.EARNINGS,
            catalyst_confidence=0.80,
            risk_level=RiskLevel.MEDIUM,
            context_summary="test",
            recommendation="valid",
        )
        assert high_conf_result.is_actionable(min_confidence=0.7)

        low_conf_result = ClaudeAnalysisResult(
            catalyst_type=CatalystType.EARNINGS,
            catalyst_confidence=0.50,
            risk_level=RiskLevel.HIGH,
            context_summary="test",
            recommendation="skip",
        )
        assert not low_conf_result.is_actionable(min_confidence=0.7)
