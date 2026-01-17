# tests/validators/test_integration.py
"""Integration tests for the complete validation pipeline."""

from unittest.mock import Mock
import pandas as pd
import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.models.social_message import SocialMessage
from src.validators import (
    TechnicalValidator,
    ValidationStatus,
    OptionsFlowData,
)


class TestValidationIntegration:
    """Integration tests for end-to-end validation pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock Alpaca client
        self.mock_alpaca = Mock()

        # Create sample OHLC data (uptrend with good indicators)
        prices = [100 + i * 0.5 for i in range(50)]  # Uptrend
        self.mock_bars_df = pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * 50,
        })

    def test_bullish_signal_passes_with_good_indicators(self):
        """End-to-end test: Bullish signal with favorable indicators passes."""
        # Arrange
        validator = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=True,
        )

        # Create realistic uptrend with pullbacks (keeps RSI in healthy 50-65 range)
        import numpy as np
        np.random.seed(42)
        base_trend = np.linspace(100, 110, 50)
        noise = np.random.normal(0, 0.5, 50)
        prices = base_trend + noise

        # Add some pullbacks to prevent overbought
        for i in [10, 20, 30, 40]:
            prices[i:i+3] -= 1.0

        self.mock_alpaca.get_bars.return_value = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1000000] * 50,
        })

        analyzed_message = AnalyzedMessage(
            message=SocialMessage(
                source="twitter",
                source_id="123",
                content="AAPL breaking out! Strong buy signal.",
                author="trader123",
                timestamp="2024-01-15T10:30:00Z",
                url="https://twitter.com/trader123/status/123",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.85,
                confidence=0.9,
            ),
        )

        # Act
        result = validator.validate(
            analyzed_message=analyzed_message,
            symbol="AAPL",
        )

        # Assert
        assert result.should_trade()
        assert result.validation.status in [ValidationStatus.PASS, ValidationStatus.WARN]
        assert result.message == analyzed_message

    def test_bearish_signal_with_options_confirmation(self):
        """End-to-end test: Bearish signal enhanced by options data."""
        # Arrange
        validator = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=True,
        )

        # Create downtrend OHLC data
        prices = list(range(150, 100, -1))  # Strong downtrend
        self.mock_alpaca.get_bars.return_value = pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * 50,
        })

        analyzed_message = AnalyzedMessage(
            message=SocialMessage(
                source="stocktwits",
                source_id="456",
                content="TSLA looking weak, major resistance ahead.",
                author="bear_trader",
                timestamp="2024-01-15T11:00:00Z",
                url="https://stocktwits.com/bear_trader/message/456",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BEARISH,
                score=0.80,
                confidence=0.85,
            ),
        )

        # Options show high put volume and unusual activity
        options_data = OptionsFlowData(
            volume_ratio=4.0,  # 400% of average - huge spike
            iv_rank=45.0,  # Low IV - cheap options
            put_call_ratio=2.5,  # High put/call ratio
            unusual_activity=True,
        )

        # Act
        result = validator.validate(
            analyzed_message=analyzed_message,
            symbol="TSLA",
            options_data=options_data,
        )

        # Assert
        assert result.validation.options_flow == options_data
        # Confidence should be boosted by volume spike and unusual activity
        assert result.validation.confidence_modifier > 1.0

    def test_veto_mode_blocks_conflicting_signals(self):
        """End-to-end test: Veto mode blocks signals with indicator conflicts."""
        # Arrange
        validator = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=True,  # Enable veto mode
        )

        # Create sideways market (low ADX, should veto)
        prices = [100] * 50  # Flat prices
        self.mock_alpaca.get_bars.return_value = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 50,
        })

        analyzed_message = AnalyzedMessage(
            message=SocialMessage(
                source="reddit",
                source_id="789",
                content="AMD to the moon!",
                author="wsb_user",
                timestamp="2024-01-15T12:00:00Z",
                url="https://reddit.com/r/wallstreetbets/comments/789",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.90,
                confidence=0.95,
            ),
        )

        # Act
        result = validator.validate(
            analyzed_message=analyzed_message,
            symbol="AMD",
        )

        # Assert
        # Should be vetoed due to weak trend (low ADX)
        assert not result.should_trade()
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert any("ADX" in reason for reason in result.validation.veto_reasons)

    def test_warn_mode_allows_signals_with_warnings(self):
        """End-to-end test: Warn mode allows signals but adds warnings."""
        # Arrange
        validator = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=False,  # Disable veto mode - use warnings instead
        )

        # Create sideways market (would normally veto)
        prices = [100] * 50
        self.mock_alpaca.get_bars.return_value = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 50,
        })

        analyzed_message = AnalyzedMessage(
            message=SocialMessage(
                source="twitter",
                source_id="999",
                content="NVDA bullish setup forming.",
                author="tech_trader",
                timestamp="2024-01-15T13:00:00Z",
                url="https://twitter.com/tech_trader/status/999",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.75,
                confidence=0.80,
            ),
        )

        # Act
        result = validator.validate(
            analyzed_message=analyzed_message,
            symbol="NVDA",
        )

        # Assert
        # Should allow trade with warnings
        assert result.should_trade()
        assert result.validation.status == ValidationStatus.WARN
        assert len(result.validation.warnings) > 0

    def test_neutral_sentiment_bypasses_all_veto_rules(self):
        """End-to-end test: Neutral sentiment bypasses veto logic entirely."""
        # Arrange
        validator = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=True,
        )

        # Create terrible market conditions (would veto bullish/bearish)
        prices = [100] * 50  # Flat
        self.mock_alpaca.get_bars.return_value = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 50,
        })

        analyzed_message = AnalyzedMessage(
            message=SocialMessage(
                source="twitter",
                source_id="111",
                content="Market is choppy, staying flat.",
                author="neutral_trader",
                timestamp="2024-01-15T14:00:00Z",
                url="https://twitter.com/neutral_trader/status/111",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.50,
                confidence=0.90,
            ),
        )

        # Act
        result = validator.validate(
            analyzed_message=analyzed_message,
            symbol="SPY",
        )

        # Assert
        # Neutral sentiment should bypass all veto rules
        assert result.should_trade()
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.veto_reasons) == 0
