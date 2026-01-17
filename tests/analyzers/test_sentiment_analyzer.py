# tests/analyzers/test_sentiment_analyzer.py
import pytest
from unittest.mock import MagicMock, patch
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.sentiment_result import SentimentLabel


class TestSentimentAnalyzer:
    @pytest.fixture
    def mock_pipeline(self):
        """Mock the transformers pipeline to avoid loading actual model."""
        with patch("src.analyzers.sentiment_analyzer.pipeline") as mock:
            mock_model = MagicMock()
            mock.return_value = mock_model
            yield mock_model

    def test_init_loads_model(self, mock_pipeline):
        with patch("src.analyzers.sentiment_analyzer.pipeline") as mock_pipe:
            mock_pipe.return_value = mock_pipeline
            analyzer = SentimentAnalyzer()
            mock_pipe.assert_called_once()

    def test_analyze_bullish_message(self, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "Bullish", "score": 0.92},
            {"label": "Bearish", "score": 0.05},
            {"label": "Neutral", "score": 0.03},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("🚨 Large $NVDA call sweep, very bullish!")

        assert result.label == SentimentLabel.BULLISH
        assert result.score == pytest.approx(0.92, rel=0.01)

    def test_analyze_bearish_message(self, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "Bullish", "score": 0.10},
            {"label": "Bearish", "score": 0.85},
            {"label": "Neutral", "score": 0.05},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("$AAPL looking weak, expecting pullback")

        assert result.label == SentimentLabel.BEARISH

    def test_analyze_batch(self, mock_pipeline):
        mock_pipeline.return_value = [
            [{"label": "Bullish", "score": 0.90}, {"label": "Bearish", "score": 0.05}, {"label": "Neutral", "score": 0.05}],
            [{"label": "Bearish", "score": 0.80}, {"label": "Bullish", "score": 0.10}, {"label": "Neutral", "score": 0.10}],
        ]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch(["bullish msg", "bearish msg"])

        assert len(results) == 2
        assert results[0].label == SentimentLabel.BULLISH
        assert results[1].label == SentimentLabel.BEARISH

    def test_analyze_empty_text_returns_neutral(self, mock_pipeline):
        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("")

        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
