# tests/integration/test_scoring_pipeline.py
"""Integration tests for the full Phase 3 -> Phase 4 scoring pipeline."""

import pytest
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from src.models.social_message import SocialMessage, SourceType
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.analyzed_message import AnalyzedMessage
from src.validators.models import (
    ValidatedSignal,
    TechnicalValidation,
    TechnicalIndicators,
    ValidationStatus,
)
from src.scoring import (
    SignalScorer,
    SourceCredibilityManager,
    TimeFactorCalculator,
    ConfluenceDetector,
    DynamicWeightCalculator,
    RecommendationBuilder,
    Direction,
    ScoreTier,
)


def create_validated_signal(
    symbol: str = "NVDA",
    sentiment_label: SentimentLabel = SentimentLabel.BULLISH,
    sentiment_score: float = 0.85,
    sentiment_confidence: float = 0.90,
    validation_status: ValidationStatus = ValidationStatus.PASS,
    rsi: float = 55.0,
    adx: float = 25.0,
    author: str = "test_user",
    timestamp: datetime = None,
    confidence_modifier: float = 1.0,
    source: SourceType = SourceType.TWITTER,
) -> ValidatedSignal:
    """Create a ValidatedSignal for testing.

    Args:
        symbol: Stock ticker symbol.
        sentiment_label: Sentiment label (BULLISH, BEARISH, NEUTRAL).
        sentiment_score: Sentiment score (0-1).
        sentiment_confidence: Sentiment confidence (0-1).
        validation_status: Technical validation status.
        rsi: RSI value (0-100).
        adx: ADX value (0-100).
        author: Author username.
        timestamp: Signal timestamp.
        confidence_modifier: Technical confidence modifier.
        source: Source type (TWITTER, REDDIT, etc).

    Returns:
        A ValidatedSignal instance for testing.
    """
    if timestamp is None:
        # Default to market hours (2:00 PM ET on a weekday)
        timestamp = datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc)

    social_message = SocialMessage(
        source=source,
        source_id=f"{source.value}_{symbol}_123",
        author=author,
        content=f"${symbol} looking great! Strong momentum here.",
        timestamp=timestamp,
        url=f"https://example.com/{source.value}/{symbol}",
        extracted_tickers=[symbol],
    )

    sentiment_result = SentimentResult(
        label=sentiment_label,
        score=sentiment_score,
        confidence=sentiment_confidence,
    )

    analyzed_message = AnalyzedMessage(
        message=social_message,
        sentiment=sentiment_result,
    )

    technical_indicators = TechnicalIndicators(
        rsi=rsi,
        macd_histogram=0.5,
        macd_trend="rising",
        stochastic_k=55.0,
        stochastic_d=50.0,
        adx=adx,
    )

    veto_reasons = []
    warnings = []
    if validation_status == ValidationStatus.VETO:
        veto_reasons = ["Technical veto: unfavorable conditions"]
    elif validation_status == ValidationStatus.WARN:
        warnings = ["Technical warning: proceed with caution"]

    technical_validation = TechnicalValidation(
        status=validation_status,
        indicators=technical_indicators,
        veto_reasons=veto_reasons,
        warnings=warnings,
        confidence_modifier=confidence_modifier,
    )

    return ValidatedSignal(
        message=analyzed_message,
        validation=technical_validation,
    )


class TestScoringPipelineIntegration:
    """Integration tests for Phase 3 -> Phase 4 scoring pipeline.

    Tests the full flow:
    1. ValidatedSignal (from Phase 3) with technical validation
    2. SignalScorer (Phase 4) calculates final score
    3. Returns TradeRecommendation with trading levels
    """

    @pytest.fixture
    def scorer(self):
        """Create a SignalScorer with real component instances."""
        credibility_manager = SourceCredibilityManager()
        time_calculator = TimeFactorCalculator()
        confluence_detector = ConfluenceDetector()
        weight_calculator = DynamicWeightCalculator()
        recommendation_builder = RecommendationBuilder()

        return SignalScorer(
            credibility_manager=credibility_manager,
            time_calculator=time_calculator,
            confluence_detector=confluence_detector,
            weight_calculator=weight_calculator,
            recommendation_builder=recommendation_builder,
        )

    @pytest.fixture
    def fresh_scorer(self):
        """Create a fresh SignalScorer with new confluence detector for isolated tests."""
        credibility_manager = SourceCredibilityManager()
        time_calculator = TimeFactorCalculator()
        confluence_detector = ConfluenceDetector()
        weight_calculator = DynamicWeightCalculator()
        recommendation_builder = RecommendationBuilder()

        return SignalScorer(
            credibility_manager=credibility_manager,
            time_calculator=time_calculator,
            confluence_detector=confluence_detector,
            weight_calculator=weight_calculator,
            recommendation_builder=recommendation_builder,
        )

    def test_full_pipeline_strong_signal(self, fresh_scorer):
        """ValidatedSignal with high sentiment + good technicals -> STRONG recommendation.

        Scenario:
        - High sentiment score (0.92)
        - PASS validation status
        - Normal RSI (55), ADX 25 (normal trend)
        - Market hours timestamp
        - Tier 1 source (unusual_whales)
        - Expected: score ~85-95, STRONG tier
        """
        # Arrange - Create a strong validated signal
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_label=SentimentLabel.BULLISH,
            sentiment_score=0.92,
            sentiment_confidence=0.90,
            validation_status=ValidationStatus.PASS,
            rsi=55.0,
            adx=25.0,
            author="unusual_whales",  # Tier 1 source
            confidence_modifier=1.0,
            # 2:00 PM ET (market hours) on Monday Jan 15, 2024
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Should be a STRONG recommendation
        assert recommendation.symbol == "NVDA"
        assert recommendation.direction == Direction.LONG
        assert recommendation.tier == ScoreTier.STRONG
        assert recommendation.score >= 80.0
        assert recommendation.score <= 100.0

        # Verify components
        assert recommendation.components.sentiment_score == 92.0  # 0.92 * 100
        assert recommendation.components.technical_score == 75.0  # PASS = 75 base
        assert recommendation.components.credibility_multiplier == 1.2  # Tier 1
        assert recommendation.components.time_factor == 1.0  # Market hours

        # Verify position size for STRONG tier
        assert recommendation.position_size_percent == 100.0

    def test_full_pipeline_moderate_signal(self, fresh_scorer):
        """ValidatedSignal with moderate confidence -> MODERATE recommendation.

        Scenario:
        - Moderate sentiment score (0.78)
        - PASS validation status
        - Expected: score ~60-75, MODERATE tier

        Score calculation:
        - sentiment_score = 78
        - technical_score = 75 (PASS base)
        - weights = 0.5 / 0.5 (ADX = 25, normal trend)
        - base_score = 78 * 0.5 + 75 * 0.5 = 76.5
        - credibility = 0.8 (Tier 3)
        - time_factor = 1.0
        - final_score = 76.5 * 0.8 * 1.0 = 61.2
        """
        # Arrange - Create a moderate validated signal
        signal = create_validated_signal(
            symbol="AAPL",
            sentiment_label=SentimentLabel.BULLISH,
            sentiment_score=0.78,  # Higher to compensate for Tier 3 penalty
            sentiment_confidence=0.75,
            validation_status=ValidationStatus.PASS,
            rsi=55.0,
            adx=25.0,
            author="random_user",  # Tier 3 source
            confidence_modifier=1.0,
            # Market hours
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 185.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Should be MODERATE tier
        assert recommendation.symbol == "AAPL"
        assert recommendation.direction == Direction.LONG
        assert recommendation.tier == ScoreTier.MODERATE
        assert recommendation.score >= 60.0
        assert recommendation.score < 80.0

        # Verify components
        assert recommendation.components.sentiment_score == 78.0  # 0.78 * 100
        assert recommendation.components.credibility_multiplier == 0.8  # Tier 3

        # Verify position size for MODERATE tier
        assert recommendation.position_size_percent == 50.0

    def test_full_pipeline_veto_signal(self, fresh_scorer):
        """ValidatedSignal with VETO status -> NO_TRADE recommendation.

        Scenario:
        - Any sentiment (high bullish)
        - VETO validation status (technical rejection)
        - Expected: score 0, NO_TRADE tier
        """
        # Arrange - Create a vetoed signal
        signal = create_validated_signal(
            symbol="TSLA",
            sentiment_label=SentimentLabel.BULLISH,
            sentiment_score=0.95,
            sentiment_confidence=0.92,
            validation_status=ValidationStatus.VETO,
            rsi=85.0,  # Overbought
            adx=35.0,
            author="unusual_whales",  # Even tier 1 source doesn't matter
            confidence_modifier=1.0,
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 250.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Should be NO_TRADE due to VETO
        assert recommendation.symbol == "TSLA"
        assert recommendation.tier == ScoreTier.NO_TRADE
        assert recommendation.score == 0.0
        assert "vetoed" in recommendation.reasoning.lower()

        # Verify position size is 0 for NO_TRADE
        assert recommendation.position_size_percent == 0.0

    def test_confluence_boost(self):
        """Multiple signals on same ticker increase score.

        Confluence logic in the scorer:
        - get_bonus() checks how many signals are already recorded
        - 2+ signals already recorded -> 10% bonus
        - 3+ signals already recorded -> 20% bonus
        - THEN the current signal is recorded

        So for the third signal to get a bonus, signals 1 and 2 must be recorded first.
        """
        # Create fresh components for this test
        credibility_manager = SourceCredibilityManager()
        time_calculator = TimeFactorCalculator()
        confluence_detector = ConfluenceDetector(
            window_minutes=15,
            bonus_2_signals=0.10,
            bonus_3_signals=0.20,
        )
        weight_calculator = DynamicWeightCalculator()
        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=credibility_manager,
            time_calculator=time_calculator,
            confluence_detector=confluence_detector,
            weight_calculator=weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        base_timestamp = datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc)

        # Create three signals on same ticker within window
        signal1 = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            author="user1",
            timestamp=base_timestamp,
        )

        signal2 = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            author="user2",
            timestamp=base_timestamp + timedelta(minutes=3),
        )

        signal3 = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            author="user3",
            timestamp=base_timestamp + timedelta(minutes=6),
        )

        current_price = 500.0

        # Act - Score all three signals
        rec1 = scorer.score(signal1, current_price)
        rec2 = scorer.score(signal2, current_price)
        rec3 = scorer.score(signal3, current_price)

        # Assert - Third signal gets confluence bonus (2 signals already recorded)
        assert rec1.components.confluence_bonus == 0.0  # No signals before
        assert rec2.components.confluence_bonus == 0.0  # Only 1 signal before (need 2)
        assert rec3.components.confluence_bonus == 0.10  # 2 signals before -> bonus

        # Third score should be higher due to confluence
        assert rec3.score > rec1.score
        assert rec3.score > rec2.score

    def test_credibility_affects_score(self, fresh_scorer):
        """Tier 1 source gets higher score than Tier 3.

        Scenario:
        - Same signal content, same technical validation
        - Different authors: unusual_whales (Tier 1) vs random_user (Tier 3)
        - Expected: Tier 1 gets higher score due to 1.2x vs 0.8x multiplier
        """
        # Create two identical signals with different authors
        base_timestamp = datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc)

        signal_tier1 = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            author="unusual_whales",  # Tier 1
            timestamp=base_timestamp,
        )

        # Create a fresh scorer for second signal to avoid confluence
        credibility_manager = SourceCredibilityManager()
        time_calculator = TimeFactorCalculator()
        confluence_detector = ConfluenceDetector()
        weight_calculator = DynamicWeightCalculator()
        recommendation_builder = RecommendationBuilder()

        scorer2 = SignalScorer(
            credibility_manager=credibility_manager,
            time_calculator=time_calculator,
            confluence_detector=confluence_detector,
            weight_calculator=weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        signal_tier3 = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            author="random_user",  # Tier 3
            timestamp=base_timestamp,
        )

        current_price = 500.0

        # Act
        rec_tier1 = fresh_scorer.score(signal_tier1, current_price)
        rec_tier3 = scorer2.score(signal_tier3, current_price)

        # Assert - Tier 1 should have higher score
        assert rec_tier1.components.credibility_multiplier == 1.2
        assert rec_tier3.components.credibility_multiplier == 0.8
        assert rec_tier1.score > rec_tier3.score

        # The ratio should be approximately 1.2 / 0.8 = 1.5
        ratio = rec_tier1.score / rec_tier3.score
        assert 1.4 <= ratio <= 1.6

    def test_time_factor_reduces_afterhours(self, fresh_scorer):
        """After-hours signal gets reduced score.

        Scenario:
        - Signal during after-hours (6:00 PM ET)
        - Expected: time_factor of 0.8 (afterhours penalty)
        """
        # Arrange - Create signal during after-hours
        # 6:00 PM ET = 11:00 PM UTC
        afterhours_timestamp = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)

        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            author="test_user",
            timestamp=afterhours_timestamp,
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Should have afterhours time factor penalty
        assert recommendation.components.time_factor == 0.8

        # Score should be reduced compared to market hours
        # Calculate expected reduction
        assert recommendation.score < 80.0  # Would be ~80 during market hours

    def test_dynamic_weights_strong_trend(self, fresh_scorer):
        """High ADX weights technical more heavily.

        Scenario:
        - ADX > 30 (strong trend)
        - Expected: technical weight should be 0.6, sentiment weight 0.4
        """
        # Arrange - Create signal with high ADX
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            adx=35.0,  # Strong trend (> 30)
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Strong trend weights
        assert recommendation.components.technical_weight == 0.6
        assert recommendation.components.sentiment_weight == 0.4

    def test_dynamic_weights_weak_trend(self, fresh_scorer):
        """Low ADX weights sentiment more heavily.

        Scenario:
        - ADX < 20 (weak trend)
        - Expected: sentiment weight should be 0.6, technical weight 0.4
        """
        # Arrange - Create signal with low ADX
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            adx=15.0,  # Weak trend (< 20)
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Weak trend weights (favor sentiment)
        assert recommendation.components.sentiment_weight == 0.6
        assert recommendation.components.technical_weight == 0.4

    def test_recommendation_has_valid_levels(self, fresh_scorer):
        """Entry, stop loss, take profit are correctly calculated.

        Scenario:
        - LONG direction
        - Entry = current price
        - Expected: stop_loss < entry < take_profit
        """
        # Arrange
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_label=SentimentLabel.BULLISH,
            sentiment_score=0.85,
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Verify LONG direction levels
        assert recommendation.direction == Direction.LONG
        assert recommendation.entry_price == current_price
        assert recommendation.stop_loss < recommendation.entry_price
        assert recommendation.take_profit > recommendation.entry_price

        # Verify risk/reward ratio
        risk = recommendation.entry_price - recommendation.stop_loss
        reward = recommendation.take_profit - recommendation.entry_price
        calculated_rr = reward / risk
        assert abs(calculated_rr - recommendation.risk_reward_ratio) < 0.01

    def test_recommendation_levels_short_direction(self, fresh_scorer):
        """Entry, stop loss, take profit for SHORT direction.

        Scenario:
        - BEARISH sentiment -> SHORT direction
        - Expected: stop_loss > entry > take_profit
        """
        # Arrange
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_label=SentimentLabel.BEARISH,
            sentiment_score=0.15,  # Low score = bearish
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Verify SHORT direction levels
        assert recommendation.direction == Direction.SHORT
        assert recommendation.entry_price == current_price
        assert recommendation.stop_loss > recommendation.entry_price
        assert recommendation.take_profit < recommendation.entry_price

    def test_weekend_time_factor_penalty(self, fresh_scorer):
        """Weekend signals get reduced score.

        Scenario:
        - Signal on Saturday
        - Expected: time_factor of 0.5 (weekend penalty)
        """
        # Arrange - Saturday Jan 20, 2024 at 2:00 PM UTC
        weekend_timestamp = datetime(2024, 1, 20, 14, 0, 0, tzinfo=timezone.utc)

        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            timestamp=weekend_timestamp,
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Should have weekend time factor penalty
        assert recommendation.components.time_factor == 0.5

    def test_premarket_time_factor(self, fresh_scorer):
        """Pre-market signals get slightly reduced score.

        Scenario:
        - Signal during pre-market (8:00 AM ET = 1:00 PM UTC)
        - Expected: time_factor of 0.9 (premarket penalty)
        """
        # Arrange - 8:00 AM ET = 1:00 PM UTC on weekday
        premarket_timestamp = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_score=0.80,
            timestamp=premarket_timestamp,
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Should have premarket time factor
        assert recommendation.components.time_factor == 0.9

    def test_warn_status_moderate_technical_score(self, fresh_scorer):
        """WARN status produces moderate technical score.

        Scenario:
        - WARN validation status
        - Expected: technical_score of 50 (moderate)
        """
        # Arrange
        signal = create_validated_signal(
            symbol="NVDA",
            validation_status=ValidationStatus.WARN,
            sentiment_score=0.80,
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert
        assert recommendation.components.technical_score == 50.0
        # Should still be tradeable (not VETO)
        assert recommendation.tier != ScoreTier.NO_TRADE

    def test_confidence_modifier_boosts_technical_score(self, fresh_scorer):
        """Confidence modifier above 1.0 boosts technical score.

        Scenario:
        - PASS status with confidence_modifier of 1.3
        - Expected: technical_score > 75 (base PASS score)
        """
        # Arrange
        signal = create_validated_signal(
            symbol="NVDA",
            validation_status=ValidationStatus.PASS,
            confidence_modifier=1.3,  # 30% boost
            sentiment_score=0.80,
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Technical score should be boosted
        # Base PASS = 75, with 1.3 modifier: 75 + (1.3 - 1.0) * 50 = 75 + 15 = 90
        assert recommendation.components.technical_score == 90.0

    def test_neutral_sentiment_direction(self, fresh_scorer):
        """Neutral sentiment produces NEUTRAL direction.

        Scenario:
        - NEUTRAL sentiment label
        - Expected: NEUTRAL direction
        """
        # Arrange
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_label=SentimentLabel.NEUTRAL,
            sentiment_score=0.50,
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
        )

        current_price = 500.0

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert
        assert recommendation.direction == Direction.NEUTRAL

    def test_triple_confluence_bonus(self):
        """Four signals on same ticker get maximum confluence bonus.

        Confluence logic:
        - Signal 1: 0 signals before -> no bonus
        - Signal 2: 1 signal before -> no bonus (need 2)
        - Signal 3: 2 signals before -> 10% bonus
        - Signal 4: 3 signals before -> 20% bonus (max)
        """
        # Create fresh components for this test
        credibility_manager = SourceCredibilityManager()
        time_calculator = TimeFactorCalculator()
        confluence_detector = ConfluenceDetector(
            window_minutes=15,
            bonus_2_signals=0.10,
            bonus_3_signals=0.20,
        )
        weight_calculator = DynamicWeightCalculator()
        recommendation_builder = RecommendationBuilder()

        scorer = SignalScorer(
            credibility_manager=credibility_manager,
            time_calculator=time_calculator,
            confluence_detector=confluence_detector,
            weight_calculator=weight_calculator,
            recommendation_builder=recommendation_builder,
        )

        base_timestamp = datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc)

        # Create 4 signals to test all bonus levels
        signals = [
            create_validated_signal(
                symbol="NVDA",
                sentiment_score=0.80,
                author=f"user{i}",
                timestamp=base_timestamp + timedelta(minutes=i * 3),
            )
            for i in range(4)
        ]

        current_price = 500.0

        # Act - Score all signals
        recommendations = [scorer.score(s, current_price) for s in signals]

        # Assert
        assert recommendations[0].components.confluence_bonus == 0.0  # 0 before
        assert recommendations[1].components.confluence_bonus == 0.0  # 1 before
        assert recommendations[2].components.confluence_bonus == 0.10  # 2 before
        assert recommendations[3].components.confluence_bonus == 0.20  # 3 before (max)

    def test_end_to_end_realistic_scenario(self, fresh_scorer):
        """Test realistic end-to-end scenario: validated signal -> trade recommendation.

        Scenario:
        - High-quality signal from tier 1 source
        - Strong technicals (PASS with boost)
        - Market hours
        - Expected: STRONG recommendation with proper levels
        """
        # Arrange - Create a high-quality validated signal
        signal = create_validated_signal(
            symbol="NVDA",
            sentiment_label=SentimentLabel.BULLISH,
            sentiment_score=0.92,
            sentiment_confidence=0.88,
            validation_status=ValidationStatus.PASS,
            rsi=55.0,
            adx=28.0,  # Strong trend but not extreme
            author="unusual_whales",  # Tier 1 source
            confidence_modifier=1.1,  # Slight boost from options flow
            timestamp=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
            source=SourceType.TWITTER,
        )

        current_price = 495.50

        # Act
        recommendation = fresh_scorer.score(signal, current_price)

        # Assert - Comprehensive validation
        # Symbol and direction
        assert recommendation.symbol == "NVDA"
        assert recommendation.direction == Direction.LONG

        # Score tier
        assert recommendation.tier == ScoreTier.STRONG
        assert recommendation.score >= 80.0

        # Components breakdown
        assert recommendation.components.sentiment_score == 92.0
        assert recommendation.components.technical_score == 80.0  # 75 + (1.1-1.0)*50
        assert recommendation.components.credibility_multiplier == 1.2  # Tier 1
        assert recommendation.components.time_factor == 1.0  # Market hours
        assert recommendation.components.confluence_bonus == 0.0  # First signal

        # Trade levels
        assert recommendation.entry_price == 495.50
        assert recommendation.stop_loss < 495.50
        assert recommendation.take_profit > 495.50
        assert recommendation.risk_reward_ratio == 2.0

        # Position sizing
        assert recommendation.position_size_percent == 100.0  # STRONG tier

        # Reasoning should be informative
        assert "Sentiment: bullish" in recommendation.reasoning
        assert "Technical: pass" in recommendation.reasoning
        assert "Credibility: 1.20x" in recommendation.reasoning

        # Timestamp should be set
        assert recommendation.timestamp is not None
