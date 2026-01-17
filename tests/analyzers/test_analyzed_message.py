# tests/analyzers/test_analyzed_message.py
import pytest
from datetime import datetime, timezone
from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType


class TestAnalyzedMessage:
    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_sentiment(self):
        return SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )

    def test_create_with_sentiment_only(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.message == sample_message
        assert analyzed.sentiment == sample_sentiment
        assert analyzed.deep_analysis is None

    def test_create_with_deep_analysis(self, sample_message, sample_sentiment):
        deep = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            context_summary="test",
            recommendation="valid",
        )
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
            deep_analysis=deep,
        )
        assert analyzed.deep_analysis == deep

    def test_is_high_signal(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.is_high_signal(min_sentiment_confidence=0.7)

    def test_requires_deep_analysis(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.requires_deep_analysis(min_sentiment_confidence=0.7)

    def test_get_tickers(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.get_tickers() == ["NVDA"]
