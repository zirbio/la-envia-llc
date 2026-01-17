# tests/scoring/test_recommendation_builder.py
"""Tests for recommendation builder."""
import pytest
from datetime import datetime

from src.scoring.models import Direction, ScoreTier, ScoreComponents, TradeRecommendation
from src.scoring.recommendation_builder import RecommendationBuilder


class TestRecommendationBuilder:
    """Tests for RecommendationBuilder."""

    def test_tier_from_score_strong(self):
        """Test that score 85 returns STRONG tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(85)

        assert tier == ScoreTier.STRONG

    def test_tier_from_score_moderate(self):
        """Test that score 65 returns MODERATE tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(65)

        assert tier == ScoreTier.MODERATE

    def test_tier_from_score_weak(self):
        """Test that score 45 returns WEAK tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(45)

        assert tier == ScoreTier.WEAK

    def test_tier_from_score_no_trade(self):
        """Test that score 35 returns NO_TRADE tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(35)

        assert tier == ScoreTier.NO_TRADE

    def test_calculate_levels_long(self):
        """Test stop loss and take profit calculation for LONG direction."""
        builder = RecommendationBuilder(
            default_stop_loss_percent=2.0,
            default_risk_reward_ratio=2.0,
        )
        entry_price = 100.0

        entry, stop_loss, take_profit = builder.calculate_levels(
            entry_price=entry_price,
            direction=Direction.LONG,
        )

        # For LONG:
        # stop_loss = 100 * (1 - 2/100) = 100 * 0.98 = 98
        # risk = 100 - 98 = 2
        # take_profit = 100 + 2 * 2 = 104
        assert entry == 100.0
        assert stop_loss == 98.0
        assert take_profit == 104.0

    def test_calculate_levels_short(self):
        """Test stop loss and take profit calculation for SHORT direction."""
        builder = RecommendationBuilder(
            default_stop_loss_percent=2.0,
            default_risk_reward_ratio=2.0,
        )
        entry_price = 100.0

        entry, stop_loss, take_profit = builder.calculate_levels(
            entry_price=entry_price,
            direction=Direction.SHORT,
        )

        # For SHORT:
        # stop_loss = 100 * (1 + 2/100) = 100 * 1.02 = 102
        # risk = 102 - 100 = 2
        # take_profit = 100 - 2 * 2 = 96
        assert entry == 100.0
        assert stop_loss == 102.0
        assert take_profit == 96.0

    def test_build_returns_complete_recommendation(self):
        """Test that build returns a complete TradeRecommendation with all fields."""
        builder = RecommendationBuilder()
        components = ScoreComponents(
            sentiment_score=80.0,
            technical_score=75.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )

        recommendation = builder.build(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            current_price=150.0,
            components=components,
            reasoning="Strong bullish sentiment with technical confirmation",
        )

        # Verify all fields are present and correct
        assert isinstance(recommendation, TradeRecommendation)
        assert recommendation.symbol == "AAPL"
        assert recommendation.direction == Direction.LONG
        assert recommendation.score == 85.0
        assert recommendation.tier == ScoreTier.STRONG
        assert recommendation.position_size_percent == 100.0  # STRONG tier
        assert recommendation.entry_price == 150.0
        # stop_loss = 150 * (1 - 2/100) = 147
        assert recommendation.stop_loss == 147.0
        # risk = 150 - 147 = 3
        # take_profit = 150 + 3 * 2 = 156
        assert recommendation.take_profit == 156.0
        assert recommendation.risk_reward_ratio == 2.0
        assert recommendation.components == components
        assert recommendation.reasoning == "Strong bullish sentiment with technical confirmation"
        assert isinstance(recommendation.timestamp, datetime)


class TestPositionSizing:
    """Tests for position sizing based on tier."""

    def test_position_size_strong(self):
        """Test STRONG tier gets 100% position size."""
        builder = RecommendationBuilder()

        size = builder.get_position_size(ScoreTier.STRONG)

        assert size == 100.0

    def test_position_size_moderate(self):
        """Test MODERATE tier gets 50% position size."""
        builder = RecommendationBuilder()

        size = builder.get_position_size(ScoreTier.MODERATE)

        assert size == 50.0

    def test_position_size_weak(self):
        """Test WEAK tier gets 25% position size."""
        builder = RecommendationBuilder()

        size = builder.get_position_size(ScoreTier.WEAK)

        assert size == 25.0

    def test_position_size_no_trade(self):
        """Test NO_TRADE tier gets 0% position size."""
        builder = RecommendationBuilder()

        size = builder.get_position_size(ScoreTier.NO_TRADE)

        assert size == 0.0


class TestCustomThresholds:
    """Tests for custom tier thresholds."""

    def test_custom_tier_thresholds(self):
        """Test that custom tier thresholds work correctly."""
        builder = RecommendationBuilder(
            tier_strong_threshold=90,
            tier_moderate_threshold=70,
            tier_weak_threshold=50,
        )

        # Score 85 would be STRONG with defaults, but MODERATE with custom
        assert builder.get_tier(85) == ScoreTier.MODERATE
        assert builder.get_tier(91) == ScoreTier.STRONG
        assert builder.get_tier(55) == ScoreTier.WEAK
        assert builder.get_tier(45) == ScoreTier.NO_TRADE

    def test_custom_position_sizes(self):
        """Test that custom position sizes work correctly."""
        builder = RecommendationBuilder(
            position_size_strong=80.0,
            position_size_moderate=40.0,
            position_size_weak=20.0,
        )

        assert builder.get_position_size(ScoreTier.STRONG) == 80.0
        assert builder.get_position_size(ScoreTier.MODERATE) == 40.0
        assert builder.get_position_size(ScoreTier.WEAK) == 20.0


class TestCustomLevels:
    """Tests for custom stop loss and risk reward."""

    def test_custom_stop_loss_percent(self):
        """Test that custom stop loss percent works."""
        builder = RecommendationBuilder()

        entry, stop_loss, take_profit = builder.calculate_levels(
            entry_price=100.0,
            direction=Direction.LONG,
            stop_loss_percent=5.0,  # 5% instead of default 2%
        )

        # stop_loss = 100 * (1 - 5/100) = 95
        assert stop_loss == 95.0

    def test_custom_risk_reward_ratio(self):
        """Test that custom risk reward ratio works."""
        builder = RecommendationBuilder()

        entry, stop_loss, take_profit = builder.calculate_levels(
            entry_price=100.0,
            direction=Direction.LONG,
            stop_loss_percent=2.0,
            risk_reward_ratio=3.0,  # 3:1 instead of default 2:1
        )

        # stop_loss = 100 * 0.98 = 98
        # risk = 2
        # take_profit = 100 + 2 * 3 = 106
        assert take_profit == 106.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_boundary_score_80_is_strong(self):
        """Test that exactly 80 is STRONG tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(80)

        assert tier == ScoreTier.STRONG

    def test_boundary_score_60_is_moderate(self):
        """Test that exactly 60 is MODERATE tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(60)

        assert tier == ScoreTier.MODERATE

    def test_boundary_score_40_is_weak(self):
        """Test that exactly 40 is WEAK tier."""
        builder = RecommendationBuilder()

        tier = builder.get_tier(40)

        assert tier == ScoreTier.WEAK

    def test_neutral_direction_uses_long_calculations(self):
        """Test that NEUTRAL direction defaults to LONG calculation."""
        builder = RecommendationBuilder()

        entry, stop_loss, take_profit = builder.calculate_levels(
            entry_price=100.0,
            direction=Direction.NEUTRAL,
        )

        # Should use LONG calculation
        assert stop_loss == 98.0
        assert take_profit == 104.0
