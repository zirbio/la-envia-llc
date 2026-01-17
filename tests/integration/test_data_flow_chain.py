import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.models.social_message import SocialMessage, SourceType
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.analyzer_manager import AnalyzerManager
from src.validators.models import (
    ValidatedSignal,
    TechnicalValidation,
    ValidationStatus,
    TechnicalIndicators,
)
from src.scoring.signal_scorer import SignalScorer
from src.scoring.models import TradeRecommendation, Direction
from src.scoring.source_credibility import SourceCredibilityManager
from src.scoring.time_factors import TimeFactorCalculator
from src.scoring.confluence_detector import ConfluenceDetector
from src.scoring.dynamic_weight_calculator import DynamicWeightCalculator
from src.scoring.recommendation_builder import RecommendationBuilder


class TestDataFlowChain:
    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="integration_test_1",
            author="unusual_whales",
            content="Massive $AAPL call sweep! Very bullish signal!",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def mock_sentiment_analyzer(self):
        mock = MagicMock()
        mock.analyze.return_value = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )
        return mock

    @pytest.fixture
    def mock_technical_indicators(self):
        return TechnicalIndicators(
            rsi=65.0,
            macd_histogram=1.5,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=35.0,
        )

    @pytest.fixture
    def signal_scorer(self):
        """Create a SignalScorer with all dependencies."""
        return SignalScorer(
            credibility_manager=SourceCredibilityManager(),
            time_calculator=TimeFactorCalculator(),
            confluence_detector=ConfluenceDetector(),
            weight_calculator=DynamicWeightCalculator(),
            recommendation_builder=RecommendationBuilder(),
        )

    def test_collector_to_analyzer(self, sample_message, mock_sentiment_analyzer):
        """Social message flows to sentiment analysis."""
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        result = manager.analyze(sample_message)

        assert isinstance(result, AnalyzedMessage)
        assert result.sentiment.label == SentimentLabel.BULLISH
        mock_sentiment_analyzer.analyze.assert_called_once_with(sample_message.content)

    def test_analyzer_to_validator(
        self, sample_message, mock_sentiment_analyzer, mock_technical_indicators
    ):
        """Analyzed message flows to technical validation."""
        # Create analyzed message
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        analyzed = manager.analyze(sample_message)

        # Create validated signal
        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=mock_technical_indicators,
            confidence_modifier=1.1,
        )
        validated = ValidatedSignal(message=analyzed, validation=validation)

        assert isinstance(validated, ValidatedSignal)
        assert validated.validation.status == ValidationStatus.PASS
        assert validated.should_trade() is True

    def test_validator_to_scoring(
        self,
        sample_message,
        mock_sentiment_analyzer,
        mock_technical_indicators,
        signal_scorer,
    ):
        """Validated signal gets scored correctly."""
        # Create analyzed message
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        analyzed = manager.analyze(sample_message)

        # Create validated signal
        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=mock_technical_indicators,
            confidence_modifier=1.1,
        )
        validated = ValidatedSignal(message=analyzed, validation=validation)

        # Score the signal
        recommendation = signal_scorer.score(
            validated_signal=validated, current_price=150.0
        )

        assert isinstance(recommendation, TradeRecommendation)
        assert recommendation.symbol == "AAPL"
        assert recommendation.direction == Direction.LONG
        assert recommendation.score > 0

    def test_full_data_flow(
        self,
        sample_message,
        mock_sentiment_analyzer,
        mock_technical_indicators,
        signal_scorer,
    ):
        """Complete chain from raw message to trade recommendation."""
        # Step 1: Analyze (Phase 2)
        analyzer = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            enable_deep_analysis=False,
        )
        analyzed = analyzer.analyze(sample_message)
        assert analyzed.sentiment.confidence >= 0.7

        # Step 2: Validate (Phase 3)
        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=mock_technical_indicators,
            confidence_modifier=1.1,
        )
        validated = ValidatedSignal(message=analyzed, validation=validation)
        assert validated.should_trade() is True

        # Step 3: Score (Phase 4)
        recommendation = signal_scorer.score(
            validated_signal=validated, current_price=150.0
        )
        assert recommendation.score > 0
        assert recommendation.direction == Direction.LONG
        assert recommendation.tier.value in ["strong", "moderate", "weak", "no_trade"]
