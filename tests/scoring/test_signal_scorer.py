# tests/scoring/test_signal_scorer.py
"""Tests for SignalScorer orchestrator."""
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentLabel, SentimentResult
from src.models.social_message import SocialMessage, SourceType
from src.scoring.confluence_detector import ConfluenceDetector
from src.scoring.dynamic_weight_calculator import DynamicWeightCalculator
from src.scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation
from src.scoring.recommendation_builder import RecommendationBuilder
from src.scoring.signal_scorer import SignalScorer
from src.scoring.source_credibility import SourceCredibilityManager
from src.scoring.time_factors import TimeFactorCalculator
from src.validators.models import (
    TechnicalIndicators,
    TechnicalValidation,
    ValidatedSignal,
    ValidationStatus,
)


@pytest.fixture
def mock_credibility_manager():
    """Create a mock SourceCredibilityManager."""
    manager = Mock(spec=SourceCredibilityManager)
    manager.get_multiplier.return_value = 1.0
    return manager


@pytest.fixture
def mock_time_calculator():
    """Create a mock TimeFactorCalculator."""
    calculator = Mock(spec=TimeFactorCalculator)
    calculator.calculate_factor.return_value = (1.0, ["market_hours"])
    return calculator


@pytest.fixture
def mock_confluence_detector():
    """Create a mock ConfluenceDetector."""
    detector = Mock(spec=ConfluenceDetector)
    detector.get_bonus.return_value = 0.0
    detector.record_signal.return_value = None
    return detector


@pytest.fixture
def mock_weight_calculator():
    """Create a mock DynamicWeightCalculator."""
    calculator = Mock(spec=DynamicWeightCalculator)
    calculator.calculate_weights.return_value = (0.5, 0.5)
    return calculator


@pytest.fixture
def mock_recommendation_builder():
    """Create a mock RecommendationBuilder."""
    builder = Mock(spec=RecommendationBuilder)
    builder.build.return_value = TradeRecommendation(
        symbol="AAPL",
        direction=Direction.LONG,
        score=75.0,
        tier=ScoreTier.MODERATE,
        position_size_percent=50.0,
        entry_price=150.0,
        stop_loss=147.0,
        take_profit=156.0,
        risk_reward_ratio=2.0,
        components=ScoreComponents(
            sentiment_score=85.0,
            technical_score=75.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.0,
            credibility_multiplier=1.0,
            time_factor=1.0,
        ),
        reasoning="Test recommendation",
        timestamp=datetime.now(),
    )
    return builder


@pytest.fixture
def signal_scorer(
    mock_credibility_manager,
    mock_time_calculator,
    mock_confluence_detector,
    mock_weight_calculator,
    mock_recommendation_builder,
):
    """Create a SignalScorer with mocked dependencies."""
    return SignalScorer(
        credibility_manager=mock_credibility_manager,
        time_calculator=mock_time_calculator,
        confluence_detector=mock_confluence_detector,
        weight_calculator=mock_weight_calculator,
        recommendation_builder=mock_recommendation_builder,
    )


@pytest.fixture
def sample_social_message():
    """Create a sample SocialMessage."""
    return SocialMessage(
        source=SourceType.TWITTER,
        source_id="12345",
        author="test_author",
        content="$AAPL looking bullish!",
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_sentiment_result():
    """Create a sample SentimentResult."""
    return SentimentResult(
        label=SentimentLabel.BULLISH,
        score=0.85,
        confidence=0.9,
    )


@pytest.fixture
def sample_technical_indicators():
    """Create sample TechnicalIndicators."""
    return TechnicalIndicators(
        rsi=55.0,
        macd_histogram=0.5,
        macd_trend="rising",
        stochastic_k=60.0,
        stochastic_d=55.0,
        adx=25.0,
    )


@pytest.fixture
def sample_analyzed_message(sample_social_message, sample_sentiment_result):
    """Create a sample AnalyzedMessage."""
    return AnalyzedMessage(
        message=sample_social_message,
        sentiment=sample_sentiment_result,
    )


@pytest.fixture
def sample_validated_signal(sample_analyzed_message, sample_technical_indicators):
    """Create a sample ValidatedSignal with PASS status."""
    return ValidatedSignal(
        message=sample_analyzed_message,
        validation=TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=sample_technical_indicators,
            confidence_modifier=1.0,
        ),
    )


@pytest.fixture
def veto_validated_signal(sample_analyzed_message, sample_technical_indicators):
    """Create a ValidatedSignal with VETO status."""
    return ValidatedSignal(
        message=sample_analyzed_message,
        validation=TechnicalValidation(
            status=ValidationStatus.VETO,
            indicators=sample_technical_indicators,
            veto_reasons=["RSI divergence detected"],
            confidence_modifier=0.5,
        ),
    )


@pytest.fixture
def warn_validated_signal(sample_analyzed_message, sample_technical_indicators):
    """Create a ValidatedSignal with WARN status."""
    return ValidatedSignal(
        message=sample_analyzed_message,
        validation=TechnicalValidation(
            status=ValidationStatus.WARN,
            indicators=sample_technical_indicators,
            warnings=["Low ADX"],
            confidence_modifier=0.9,
        ),
    )


@pytest.fixture
def neutral_sentiment_result():
    """Create a neutral SentimentResult."""
    return SentimentResult(
        label=SentimentLabel.NEUTRAL,
        score=0.5,
        confidence=0.8,
    )


@pytest.fixture
def neutral_analyzed_message(sample_social_message, neutral_sentiment_result):
    """Create an AnalyzedMessage with neutral sentiment."""
    return AnalyzedMessage(
        message=sample_social_message,
        sentiment=neutral_sentiment_result,
    )


@pytest.fixture
def neutral_validated_signal(neutral_analyzed_message, sample_technical_indicators):
    """Create a ValidatedSignal with neutral sentiment."""
    return ValidatedSignal(
        message=neutral_analyzed_message,
        validation=TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=sample_technical_indicators,
            confidence_modifier=1.0,
        ),
    )


class TestSignalScorer:
    """Tests for SignalScorer class."""

    def test_score_returns_trade_recommendation(
        self, signal_scorer, sample_validated_signal
    ):
        """Verify score() returns a TradeRecommendation."""
        result = signal_scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        assert isinstance(result, TradeRecommendation)
        assert result.symbol == "AAPL"

    def test_score_applies_dynamic_weights(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        mock_recommendation_builder,
        sample_validated_signal,
    ):
        """Verify dynamic weights are calculated and used."""
        # Configure weight calculator to return specific weights
        mock_weight_calculator.calculate_weights.return_value = (0.4, 0.6)

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=mock_recommendation_builder,
        )

        scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        # Verify weight calculator was called with ADX value
        mock_weight_calculator.calculate_weights.assert_called_once()
        call_args = mock_weight_calculator.calculate_weights.call_args
        assert call_args[1]["adx"] == 25.0  # From sample_technical_indicators

    def test_score_applies_credibility_multiplier(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        mock_recommendation_builder,
        sample_validated_signal,
    ):
        """Verify credibility multiplier is applied."""
        # Set credibility multiplier to 1.2
        mock_credibility_manager.get_multiplier.return_value = 1.2

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=mock_recommendation_builder,
        )

        scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        # Verify credibility manager was called with author
        mock_credibility_manager.get_multiplier.assert_called_once_with(
            "test_author", SourceType.TWITTER
        )

    def test_score_applies_time_factor(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        mock_recommendation_builder,
        sample_validated_signal,
    ):
        """Verify time factor is applied."""
        # Set time factor to 0.9 (premarket)
        mock_time_calculator.calculate_factor.return_value = (0.9, ["premarket"])

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=mock_recommendation_builder,
        )

        scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        # Verify time calculator was called
        mock_time_calculator.calculate_factor.assert_called_once()

    def test_score_applies_confluence_bonus(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        mock_recommendation_builder,
        sample_validated_signal,
    ):
        """Verify confluence bonus is applied."""
        # Set confluence bonus to 0.1 (2 signals within window)
        mock_confluence_detector.get_bonus.return_value = 0.1

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=mock_recommendation_builder,
        )

        scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        # Verify confluence detector was called
        mock_confluence_detector.get_bonus.assert_called_once()
        mock_confluence_detector.record_signal.assert_called_once()

    def test_veto_signal_returns_no_trade(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        sample_validated_signal,
        veto_validated_signal,
    ):
        """VETO status returns NO_TRADE tier with score 0."""
        # Use a real RecommendationBuilder to verify the result
        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        result = scorer.score(
            validated_signal=veto_validated_signal,
            current_price=150.0,
        )

        assert result.tier == ScoreTier.NO_TRADE
        assert result.score == 0.0

    def test_neutral_sentiment_returns_neutral_direction(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        neutral_validated_signal,
    ):
        """NEUTRAL sentiment returns Direction.NEUTRAL."""
        # Use a real RecommendationBuilder to verify the result
        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        result = scorer.score(
            validated_signal=neutral_validated_signal,
            current_price=150.0,
        )

        assert result.direction == Direction.NEUTRAL


class TestSentimentScoreCalculation:
    """Tests for sentiment score calculation."""

    def test_bullish_sentiment_converts_to_score(
        self, signal_scorer, sample_validated_signal
    ):
        """Bullish sentiment score (0.85) converts to 85."""
        # Access private method for testing
        score = signal_scorer._calculate_sentiment_score(sample_validated_signal)
        assert score == 85.0

    def test_bearish_sentiment_converts_to_score(
        self, signal_scorer, sample_social_message, sample_technical_indicators
    ):
        """Bearish sentiment score (0.3) converts to 30."""
        sentiment = SentimentResult(
            label=SentimentLabel.BEARISH,
            score=0.3,
            confidence=0.9,
        )
        analyzed = AnalyzedMessage(
            message=sample_social_message,
            sentiment=sentiment,
        )
        signal = ValidatedSignal(
            message=analyzed,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=sample_technical_indicators,
                confidence_modifier=1.0,
            ),
        )

        score = signal_scorer._calculate_sentiment_score(signal)
        assert score == 30.0


class TestTechnicalScoreCalculation:
    """Tests for technical score calculation."""

    def test_pass_status_with_default_modifier(
        self, signal_scorer, sample_validated_signal
    ):
        """PASS with confidence_modifier=1.0 gives 75."""
        score = signal_scorer._calculate_technical_score(sample_validated_signal)
        assert score == 75.0

    def test_pass_status_with_high_modifier(
        self,
        signal_scorer,
        sample_analyzed_message,
        sample_technical_indicators,
    ):
        """PASS with confidence_modifier=1.5 gives higher score."""
        signal = ValidatedSignal(
            message=sample_analyzed_message,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=sample_technical_indicators,
                confidence_modifier=1.5,
            ),
        )

        score = signal_scorer._calculate_technical_score(signal)
        # 75 + (1.5 - 1.0) * 50 = 75 + 25 = 100
        assert score == 100.0

    def test_veto_status_gives_low_score(
        self, signal_scorer, veto_validated_signal
    ):
        """VETO status gives score of 25."""
        score = signal_scorer._calculate_technical_score(veto_validated_signal)
        assert score == 25.0

    def test_warn_status_gives_medium_score(
        self, signal_scorer, warn_validated_signal
    ):
        """WARN status gives score of 50."""
        score = signal_scorer._calculate_technical_score(warn_validated_signal)
        assert score == 50.0


class TestDirectionDetermination:
    """Tests for direction determination."""

    def test_bullish_returns_long(self, signal_scorer, sample_validated_signal):
        """BULLISH sentiment returns Direction.LONG."""
        direction = signal_scorer._determine_direction(sample_validated_signal)
        assert direction == Direction.LONG

    def test_bearish_returns_short(
        self, signal_scorer, sample_social_message, sample_technical_indicators
    ):
        """BEARISH sentiment returns Direction.SHORT."""
        sentiment = SentimentResult(
            label=SentimentLabel.BEARISH,
            score=0.2,
            confidence=0.9,
        )
        analyzed = AnalyzedMessage(
            message=sample_social_message,
            sentiment=sentiment,
        )
        signal = ValidatedSignal(
            message=analyzed,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=sample_technical_indicators,
                confidence_modifier=1.0,
            ),
        )

        direction = signal_scorer._determine_direction(signal)
        assert direction == Direction.SHORT

    def test_neutral_returns_neutral(
        self, signal_scorer, neutral_validated_signal
    ):
        """NEUTRAL sentiment returns Direction.NEUTRAL."""
        direction = signal_scorer._determine_direction(neutral_validated_signal)
        assert direction == Direction.NEUTRAL


class TestScoringFormula:
    """Tests for the complete scoring formula."""

    def test_scoring_formula_calculation(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        sample_validated_signal,
    ):
        """Verify complete scoring formula is correctly applied."""
        # Configure mocks
        mock_weight_calculator.calculate_weights.return_value = (0.5, 0.5)
        mock_credibility_manager.get_multiplier.return_value = 1.2
        mock_time_calculator.calculate_factor.return_value = (0.9, ["premarket"])
        mock_confluence_detector.get_bonus.return_value = 0.1

        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        result = scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        # Calculate expected score:
        # sentiment_score = 85 (0.85 * 100)
        # technical_score = 75 (PASS with modifier 1.0)
        # base_score = (85 * 0.5) + (75 * 0.5) = 42.5 + 37.5 = 80.0
        # final_score = 80.0 * 1.2 * 0.9 * (1 + 0.1) = 80.0 * 1.188 = 95.04
        expected_score = 80.0 * 1.2 * 0.9 * 1.1
        assert abs(result.score - expected_score) < 0.01

    def test_score_clamped_to_100(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        sample_validated_signal,
    ):
        """Score is clamped to maximum of 100."""
        # Configure very high multipliers to exceed 100
        mock_weight_calculator.calculate_weights.return_value = (0.5, 0.5)
        mock_credibility_manager.get_multiplier.return_value = 1.5
        mock_time_calculator.calculate_factor.return_value = (1.0, ["market_hours"])
        mock_confluence_detector.get_bonus.return_value = 0.2

        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        result = scorer.score(
            validated_signal=sample_validated_signal,
            current_price=150.0,
        )

        assert result.score <= 100.0

    def test_score_clamped_to_0(
        self,
        mock_credibility_manager,
        mock_time_calculator,
        mock_confluence_detector,
        mock_weight_calculator,
        sample_social_message,
        sample_technical_indicators,
    ):
        """Score is clamped to minimum of 0."""
        # Very low sentiment and technical scores
        sentiment = SentimentResult(
            label=SentimentLabel.BEARISH,
            score=0.1,
            confidence=0.5,
        )
        analyzed = AnalyzedMessage(
            message=sample_social_message,
            sentiment=sentiment,
        )
        signal = ValidatedSignal(
            message=analyzed,
            validation=TechnicalValidation(
                status=ValidationStatus.VETO,
                indicators=sample_technical_indicators,
                veto_reasons=["Multiple veto reasons"],
                confidence_modifier=0.1,
            ),
        )

        # Configure low multipliers
        mock_weight_calculator.calculate_weights.return_value = (0.5, 0.5)
        mock_credibility_manager.get_multiplier.return_value = 0.8
        mock_time_calculator.calculate_factor.return_value = (0.5, ["weekend"])
        mock_confluence_detector.get_bonus.return_value = 0.0

        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=mock_credibility_manager,
            time_calculator=mock_time_calculator,
            confluence_detector=mock_confluence_detector,
            weight_calculator=mock_weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        result = scorer.score(
            validated_signal=signal,
            current_price=150.0,
        )

        # VETO always returns 0
        assert result.score == 0.0
