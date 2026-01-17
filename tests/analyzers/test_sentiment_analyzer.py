# tests/analyzers/test_sentiment_analyzer.py
import pytest
from unittest.mock import MagicMock, patch
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.sentiment_result import SentimentLabel


class TestSentimentAnalyzer:
    def test_init_loads_model(self):
        with patch("src.analyzers.sentiment_analyzer.pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock()
            analyzer = SentimentAnalyzer()
            mock_pipe.assert_called_once()

    def test_analyze_bullish_message(self):
        mock_model = MagicMock()
        mock_model.return_value = [[
            {"label": "Bullish", "score": 0.92},
            {"label": "Bearish", "score": 0.05},
            {"label": "Neutral", "score": 0.03},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_model):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("Large $NVDA call sweep, very bullish!")

        assert result.label == SentimentLabel.BULLISH
        assert result.score == pytest.approx(0.92, rel=0.01)
        assert result.confidence == pytest.approx(0.92, rel=0.01)

    def test_analyze_bearish_message(self):
        mock_model = MagicMock()
        mock_model.return_value = [[
            {"label": "Bullish", "score": 0.10},
            {"label": "Bearish", "score": 0.85},
            {"label": "Neutral", "score": 0.05},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_model):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("$AAPL looking weak, expecting pullback")

        assert result.label == SentimentLabel.BEARISH
        # Strong bearish (0.85) should result in low score (0.15)
        assert result.score == pytest.approx(0.15, rel=0.01)
        assert result.confidence == pytest.approx(0.85, rel=0.01)

    def test_analyze_neutral_dominant_message(self):
        mock_model = MagicMock()
        mock_model.return_value = [[
            {"label": "Bullish", "score": 0.20},
            {"label": "Bearish", "score": 0.25},
            {"label": "Neutral", "score": 0.55},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_model):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("$TSLA earnings report tomorrow")

        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == pytest.approx(0.5, rel=0.01)
        assert result.confidence == pytest.approx(0.55, rel=0.01)

    def test_analyze_batch(self):
        mock_model = MagicMock()
        mock_model.return_value = [
            [{"label": "Bullish", "score": 0.90}, {"label": "Bearish", "score": 0.05}, {"label": "Neutral", "score": 0.05}],
            [{"label": "Bearish", "score": 0.80}, {"label": "Bullish", "score": 0.10}, {"label": "Neutral", "score": 0.10}],
        ]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_model):
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch(["bullish msg", "bearish msg"])

        assert len(results) == 2
        assert results[0].label == SentimentLabel.BULLISH
        assert results[1].label == SentimentLabel.BEARISH

    def test_analyze_empty_text_returns_neutral(self):
        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=MagicMock()):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("")

        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_analyze_whitespace_only_text_returns_neutral(self):
        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=MagicMock()):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("   ")

        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_analyze_batch_with_empty_and_whitespace(self):
        mock_model = MagicMock()
        # Only one valid text, so only one result from pipeline
        mock_model.return_value = [
            [{"label": "Bullish", "score": 0.90}, {"label": "Bearish", "score": 0.05}, {"label": "Neutral", "score": 0.05}],
        ]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_model):
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch(["", "valid text", "   "])

        assert len(results) == 3
        # Empty and whitespace texts should be neutral with 0 confidence
        assert results[0].label == SentimentLabel.NEUTRAL
        assert results[0].confidence == 0.0
        assert results[1].label == SentimentLabel.BULLISH
        assert results[2].label == SentimentLabel.NEUTRAL
        assert results[2].confidence == 0.0
