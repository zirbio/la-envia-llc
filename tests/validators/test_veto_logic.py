# tests/validators/test_veto_logic.py
"""Tests for veto logic rules."""


from src.analyzers.sentiment_result import SentimentLabel
from src.validators.models import TechnicalIndicators, ValidationStatus
from src.validators.veto_logic import VetoLogic


class TestVetoLogic:
    """Test suite for VetoLogic class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.veto_logic = VetoLogic(
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
        )

    def test_bullish_signal_vetoed_when_overbought(self):
        """BULLISH signal should be vetoed when RSI > 70."""
        indicators = TechnicalIndicators(
            rsi=75.0,  # Overbought
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any(
            "RSI" in reason and "overbought" in reason.lower()
            for reason in reasons
        )

    def test_bearish_signal_vetoed_when_oversold(self):
        """BEARISH signal should be vetoed when RSI < 30."""
        indicators = TechnicalIndicators(
            rsi=25.0,  # Oversold
            macd_histogram=-0.5,
            macd_trend="falling",
            stochastic_k=20.0,
            stochastic_d=25.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BEARISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any(
            "RSI" in reason and "oversold" in reason.lower()
            for reason in reasons
        )

    def test_bullish_signal_passes_normal_rsi(self):
        """BULLISH signal should pass with normal RSI range."""
        indicators = TechnicalIndicators(
            rsi=55.0,  # Normal range
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=60.0,
            stochastic_d=55.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.PASS
        assert len(reasons) == 0

    def test_signal_vetoed_when_no_trend(self):
        """Any signal should be vetoed when ADX < threshold (no trend)."""
        indicators = TechnicalIndicators(
            rsi=55.0,
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=60.0,
            stochastic_d=55.0,
            adx=15.0,  # Below threshold
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any("ADX" in reason and "trend" in reason.lower() for reason in reasons)

    def test_bullish_vetoed_with_bearish_macd_divergence(self):
        """BULLISH signal should be vetoed with bearish MACD divergence."""
        indicators = TechnicalIndicators(
            rsi=55.0,
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=60.0,
            stochastic_d=55.0,
            adx=25.0,
            macd_divergence="bearish",
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any(
            "MACD divergence" in reason and "bearish" in reason.lower()
            for reason in reasons
        )

    def test_bearish_vetoed_with_bullish_macd_divergence(self):
        """BEARISH signal should be vetoed with bullish MACD divergence."""
        indicators = TechnicalIndicators(
            rsi=45.0,
            macd_histogram=-0.5,
            macd_trend="falling",
            stochastic_k=40.0,
            stochastic_d=45.0,
            adx=25.0,
            macd_divergence="bullish",
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BEARISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any(
            "MACD divergence" in reason and "bullish" in reason.lower()
            for reason in reasons
        )

    def test_bullish_vetoed_with_falling_macd_and_negative_histogram(self):
        """BULLISH signal should be vetoed if MACD is falling and histogram < 0."""
        indicators = TechnicalIndicators(
            rsi=55.0,
            macd_histogram=-0.2,  # Negative
            macd_trend="falling",  # Falling
            stochastic_k=60.0,
            stochastic_d=55.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any(
            "MACD" in reason and "falling" in reason.lower()
            for reason in reasons
        )

    def test_bearish_vetoed_with_rising_macd_and_positive_histogram(self):
        """BEARISH signal should be vetoed if MACD is rising and histogram > 0."""
        indicators = TechnicalIndicators(
            rsi=45.0,
            macd_histogram=0.2,  # Positive
            macd_trend="rising",  # Rising
            stochastic_k=40.0,
            stochastic_d=45.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BEARISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) > 0
        assert any(
            "MACD" in reason and "rising" in reason.lower()
            for reason in reasons
        )

    def test_neutral_signal_always_passes(self):
        """NEUTRAL sentiment should always pass (no veto rules apply)."""
        indicators = TechnicalIndicators(
            rsi=75.0,  # Would veto BULLISH
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=15.0,  # Would veto any signal
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.NEUTRAL)

        assert status == ValidationStatus.PASS
        assert len(reasons) == 0

    def test_bullish_passes_with_falling_macd_but_positive_histogram(self):
        """BULLISH should pass if MACD falling but histogram is positive."""
        indicators = TechnicalIndicators(
            rsi=55.0,
            macd_histogram=0.1,  # Positive
            macd_trend="falling",  # Falling - but histogram positive so no veto
            stochastic_k=60.0,
            stochastic_d=55.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.PASS
        assert len(reasons) == 0

    def test_bearish_passes_with_rising_macd_but_negative_histogram(self):
        """BEARISH should pass if MACD rising but histogram is negative."""
        indicators = TechnicalIndicators(
            rsi=45.0,
            macd_histogram=-0.1,  # Negative
            macd_trend="rising",  # Rising - but histogram negative so no veto
            stochastic_k=40.0,
            stochastic_d=45.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BEARISH)

        assert status == ValidationStatus.PASS
        assert len(reasons) == 0

    def test_multiple_veto_reasons_accumulated(self):
        """Multiple veto conditions should all be reported."""
        indicators = TechnicalIndicators(
            rsi=75.0,  # Overbought
            macd_histogram=-0.2,  # Negative
            macd_trend="falling",  # Falling
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=15.0,  # No trend
            macd_divergence="bearish",  # Bearish divergence
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.VETO
        assert len(reasons) >= 3  # Should have multiple veto reasons

    def test_custom_thresholds(self):
        """VetoLogic should respect custom threshold parameters."""
        custom_logic = VetoLogic(
            rsi_overbought=80.0,  # Higher threshold
            rsi_oversold=20.0,    # Lower threshold
            adx_trend_threshold=25.0,  # Higher threshold
        )

        indicators = TechnicalIndicators(
            rsi=75.0,  # Would be overbought with default, not with custom
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=65.0,
            adx=26.0,  # Would fail default threshold (20), passes custom (25)
        )

        status, reasons = custom_logic.evaluate(indicators, SentimentLabel.BULLISH)

        assert status == ValidationStatus.PASS
        assert len(reasons) == 0

    def test_bearish_signal_passes_normal_conditions(self):
        """BEARISH signal should pass with normal conditions."""
        indicators = TechnicalIndicators(
            rsi=45.0,  # Normal range
            macd_histogram=-0.5,
            macd_trend="falling",
            stochastic_k=40.0,
            stochastic_d=45.0,
            adx=25.0,
        )

        status, reasons = self.veto_logic.evaluate(indicators, SentimentLabel.BEARISH)

        assert status == ValidationStatus.PASS
        assert len(reasons) == 0
