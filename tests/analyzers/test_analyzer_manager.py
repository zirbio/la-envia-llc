# tests/analyzers/test_analyzer_manager.py
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from src.analyzers.analyzer_manager import AnalyzerManager
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType


class TestAnalyzerManager:
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        mock = MagicMock()
        mock.analyze.return_value = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.90,
            confidence=0.85,
        )
        mock.analyze_batch.return_value = [
            SentimentResult(label=SentimentLabel.BULLISH, score=0.90, confidence=0.85),
            SentimentResult(label=SentimentLabel.NEUTRAL, score=0.50, confidence=0.60),
        ]
        return mock

    @pytest.fixture
    def mock_claude_analyzer(self):
        mock = MagicMock()
        mock.analyze.return_value = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            context_summary="Large sweep detected",
            recommendation="valid",
        )
        return mock

    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
        )

    def test_init(self, mock_sentiment_analyzer, mock_claude_analyzer):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
        )
        assert manager.sentiment_analyzer == mock_sentiment_analyzer
        assert manager.claude_analyzer == mock_claude_analyzer

    def test_analyze_single_message(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
        )
        result = manager.analyze(sample_message)

        assert result.sentiment.label == SentimentLabel.BULLISH
        mock_sentiment_analyzer.analyze.assert_called_once_with(sample_message.content)

    def test_analyze_triggers_deep_analysis_for_high_signal(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )
        result = manager.analyze(sample_message)

        assert result.deep_analysis is not None
        mock_claude_analyzer.analyze.assert_called_once()

    def test_analyze_skips_deep_analysis_for_low_confidence(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.60,
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )
        result = manager.analyze(sample_message)

        assert result.deep_analysis is None
        mock_claude_analyzer.analyze.assert_not_called()

    def test_analyze_batch(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        messages = [sample_message, sample_message]

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            enable_deep_analysis=False,
        )
        results = manager.analyze_batch(messages)

        assert len(results) == 2
        mock_sentiment_analyzer.analyze_batch.assert_called_once()

    def test_disabled_deep_analysis(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            enable_deep_analysis=False,
        )
        result = manager.analyze(sample_message)

        assert result.deep_analysis is None
        mock_claude_analyzer.analyze.assert_not_called()

    def test_init_validates_min_sentiment_confidence_negative(
        self, mock_sentiment_analyzer
    ):
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            AnalyzerManager(
                sentiment_analyzer=mock_sentiment_analyzer,
                min_sentiment_confidence=-0.5,
            )

    def test_init_validates_min_sentiment_confidence_above_one(
        self, mock_sentiment_analyzer
    ):
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            AnalyzerManager(
                sentiment_analyzer=mock_sentiment_analyzer,
                min_sentiment_confidence=1.5,
            )

    def test_analyze_handles_sentiment_analyzer_exception(
        self, mock_sentiment_analyzer, sample_message
    ):
        mock_sentiment_analyzer.analyze.side_effect = RuntimeError("Model error")

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        result = manager.analyze(sample_message)

        # Should return neutral sentiment with zero confidence on error
        assert result.sentiment.label == SentimentLabel.NEUTRAL
        assert result.sentiment.confidence == 0.0

    def test_analyze_batch_handles_exception_with_fallback(
        self, mock_sentiment_analyzer, sample_message
    ):
        # batch fails, individual succeeds
        mock_sentiment_analyzer.analyze_batch.side_effect = RuntimeError("Batch error")
        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.90,
            confidence=0.85,
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        messages = [sample_message, sample_message]
        results = manager.analyze_batch(messages)

        # Should fall back to individual analysis
        assert len(results) == 2
        assert mock_sentiment_analyzer.analyze.call_count == 2

    def test_analyze_batch_empty_list(self, mock_sentiment_analyzer):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        results = manager.analyze_batch([])

        assert results == []
        mock_sentiment_analyzer.analyze_batch.assert_not_called()
