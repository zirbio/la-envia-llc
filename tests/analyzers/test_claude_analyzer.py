# tests/analyzers/test_claude_analyzer.py
import pytest
from unittest.mock import MagicMock, patch
import json
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.claude_result import CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType
from datetime import datetime, timezone


class TestClaudeAnalyzer:
    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep $142 strike 2/21 exp, $2.4M premium",
            timestamp=datetime.now(timezone.utc),
            url="https://twitter.com/unusual_whales/status/123",
        )

    def test_init_creates_client(self):
        with patch("src.analyzers.claude_analyzer.Anthropic") as mock:
            analyzer = ClaudeAnalyzer(api_key="test-key")
            mock.assert_called_once_with(api_key="test-key")

    def test_analyze_returns_result(self, sample_message):
        with patch("src.analyzers.claude_analyzer.Anthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=json.dumps({
                "catalyst_type": "institutional_flow",
                "catalyst_confidence": 0.85,
                "risk_level": "low",
                "risk_factors": ["earnings_in_3_weeks"],
                "context_summary": "Large call sweep indicates institutional accumulation",
                "recommendation": "valid_catalyst",
                "reasoning": "Sweep size and timing suggest informed buying",
            }))]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            analyzer = ClaudeAnalyzer(api_key="test-key")
            result = analyzer.analyze(sample_message)

            assert result.catalyst_type == CatalystType.INSTITUTIONAL_FLOW
            assert result.catalyst_confidence == 0.85
            assert result.risk_level == RiskLevel.LOW
            assert "earnings_in_3_weeks" in result.risk_factors

    def test_analyze_handles_api_error(self, sample_message):
        with patch("src.analyzers.claude_analyzer.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

            analyzer = ClaudeAnalyzer(api_key="test-key")
            result = analyzer.analyze(sample_message)

            assert result.catalyst_type == CatalystType.UNKNOWN
            assert result.risk_level == RiskLevel.HIGH

    def test_analyze_handles_malformed_json(self, sample_message):
        with patch("src.analyzers.claude_analyzer.Anthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="This is not valid JSON {broken")]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            analyzer = ClaudeAnalyzer(api_key="test-key")
            result = analyzer.analyze(sample_message)

            assert result.catalyst_type == CatalystType.UNKNOWN
            assert result.risk_level == RiskLevel.HIGH
            assert "analysis_failed" in result.risk_factors
            assert result.recommendation == "skip"

    def test_rate_limiting(self, sample_message):
        with patch("src.analyzers.claude_analyzer.Anthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=json.dumps({
                "catalyst_type": "institutional_flow",
                "catalyst_confidence": 0.85,
                "risk_level": "low",
                "risk_factors": [],
                "context_summary": "test",
                "recommendation": "valid",
            }))]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            analyzer = ClaudeAnalyzer(api_key="test-key", rate_limit_per_minute=60)
            assert analyzer.rate_limit_per_minute == 60
