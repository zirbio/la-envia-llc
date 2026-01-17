# tests/integration/test_analyzer_pipeline.py
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from src.models.social_message import SocialMessage, SourceType
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.analyzers.analyzer_manager import AnalyzerManager


class TestAnalyzerPipelineIntegration:
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_claude_analyzer(self):
        mock = MagicMock()
        return mock

    def test_full_pipeline_bullish_flow(
        self, mock_sentiment_analyzer, mock_claude_analyzer
    ):
        """Test full pipeline: bullish message → sentiment → deep analysis."""
        # Setup
        message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep $142 strike, $2.4M premium",
            timestamp=datetime.now(timezone.utc),
        )

        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )

        mock_claude_analyzer.analyze.return_value = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            risk_factors=["earnings_in_3_weeks"],
            context_summary="Large call sweep indicates institutional accumulation",
            recommendation="valid_catalyst",
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )

        # Execute
        result = manager.analyze(message)

        # Verify
        assert result.sentiment.label == SentimentLabel.BULLISH
        assert result.sentiment.confidence >= 0.7
        assert result.deep_analysis is not None
        assert result.deep_analysis.catalyst_type == CatalystType.INSTITUTIONAL_FLOW
        assert result.is_high_signal()

    def test_full_pipeline_neutral_flow(
        self, mock_sentiment_analyzer, mock_claude_analyzer
    ):
        """Test pipeline: neutral message → sentiment only, no deep analysis."""
        message = SocialMessage(
            source=SourceType.REDDIT,
            source_id="456",
            author="random_user",
            content="What do you guys think about $AAPL?",
            timestamp=datetime.now(timezone.utc),
            subreddit="stocks",
        )

        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.65,
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )

        result = manager.analyze(message)

        # Verify - should NOT trigger deep analysis
        assert result.sentiment.label == SentimentLabel.NEUTRAL
        assert result.deep_analysis is None
        mock_claude_analyzer.analyze.assert_not_called()

    def test_batch_processing(self, mock_sentiment_analyzer, mock_claude_analyzer):
        """Test batch processing of multiple messages."""
        messages = [
            SocialMessage(
                source=SourceType.TWITTER,
                source_id="1",
                author="trader1",
                content="$NVDA looking strong!",
                timestamp=datetime.now(timezone.utc),
            ),
            SocialMessage(
                source=SourceType.STOCKTWITS,
                source_id="2",
                author="trader2",
                content="$AAPL might pull back",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        mock_sentiment_analyzer.analyze_batch.return_value = [
            SentimentResult(label=SentimentLabel.BULLISH, score=0.85, confidence=0.80),
            SentimentResult(label=SentimentLabel.BEARISH, score=0.25, confidence=0.75),
        ]

        mock_claude_analyzer.analyze.return_value = ClaudeAnalysisResult(
            catalyst_type=CatalystType.TECHNICAL_BREAKOUT,
            catalyst_confidence=0.70,
            risk_level=RiskLevel.MEDIUM,
            context_summary="test",
            recommendation="valid",
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )

        results = manager.analyze_batch(messages)

        assert len(results) == 2
        # Both should have deep analysis (both above confidence threshold)
        assert results[0].deep_analysis is not None
        assert results[1].deep_analysis is not None
