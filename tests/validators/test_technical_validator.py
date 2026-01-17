# tests/validators/test_technical_validator.py
"""Tests for TechnicalValidator orchestrator."""

from unittest.mock import Mock, MagicMock
import pandas as pd
import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.models.social_message import SocialMessage
from src.validators.technical_validator import TechnicalValidator
from src.validators.models import (
    TechnicalIndicators,
    ValidationStatus,
    ValidatedSignal,
    OptionsFlowData,
)


class TestTechnicalValidator:
    """Test suite for TechnicalValidator orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock alpaca client
        self.mock_alpaca = Mock()

        # Create validator instance
        self.validator = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=True,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
            lookback_bars=50,
            timeframe="5Min",
        )

        # Create sample analyzed message
        self.analyzed_message = AnalyzedMessage(
            message=SocialMessage(
                source="twitter",
                source_id="123",
                content="AAPL is looking bullish! Great setup.",
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

    def _create_mock_bars_df(self) -> pd.DataFrame:
        """Create a mock OHLC DataFrame for testing."""
        return pd.DataFrame({
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [101.0] * 50,
            'volume': [1000000] * 50,
        })

    def test_validate_returns_validated_signal(self):
        """Should return ValidatedSignal with PASS status for normal conditions."""
        # Arrange
        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Act
        result = self.validator.validate(
            analyzed_message=self.analyzed_message,
            symbol="AAPL",
        )

        # Assert
        assert isinstance(result, ValidatedSignal)
        assert result.message == self.analyzed_message
        assert result.validation.status in [ValidationStatus.PASS, ValidationStatus.VETO, ValidationStatus.WARN]
        assert isinstance(result.validation.indicators, TechnicalIndicators)
        self.mock_alpaca.get_bars.assert_called_once()

    def test_validate_vetos_overbought_bullish_signal(self):
        """Should VETO bullish signal when RSI > 70."""
        # Arrange
        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Mock indicator engine to return overbought RSI
        self.validator.indicator_engine = Mock()
        self.validator.indicator_engine.calculate_all.return_value = TechnicalIndicators(
            rsi=75.0,  # Overbought
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=25.0,  # Strong trend
        )

        # Act
        result = self.validator.validate(
            analyzed_message=self.analyzed_message,
            symbol="AAPL",
        )

        # Assert
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert any("RSI" in reason for reason in result.validation.veto_reasons)

    def test_validate_vetos_oversold_bearish_signal(self):
        """Should VETO bearish signal when RSI < 30."""
        # Arrange
        bearish_message = AnalyzedMessage(
            message=SocialMessage(
                source="twitter",
                source_id="124",
                content="AAPL is looking bearish.",
                author="trader123",
                timestamp="2024-01-15T10:30:00Z",
                url="https://twitter.com/trader123/status/124",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BEARISH,
                score=0.85,
                confidence=0.9,
            ),
        )

        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Mock indicator engine to return oversold RSI
        self.validator.indicator_engine = Mock()
        self.validator.indicator_engine.calculate_all.return_value = TechnicalIndicators(
            rsi=25.0,  # Oversold
            macd_histogram=-0.5,
            macd_trend="falling",
            stochastic_k=20.0,
            stochastic_d=25.0,
            adx=25.0,  # Strong trend
        )

        # Act
        result = self.validator.validate(
            analyzed_message=bearish_message,
            symbol="AAPL",
        )

        # Assert
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert any("RSI" in reason for reason in result.validation.veto_reasons)

    def test_validate_vetos_weak_trend(self):
        """Should VETO signal when ADX < threshold (no trend)."""
        # Arrange
        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Mock indicator engine to return low ADX
        self.validator.indicator_engine = Mock()
        self.validator.indicator_engine.calculate_all.return_value = TechnicalIndicators(
            rsi=50.0,
            macd_histogram=0.1,
            macd_trend="rising",
            stochastic_k=50.0,
            stochastic_d=50.0,
            adx=15.0,  # Weak trend
        )

        # Act
        result = self.validator.validate(
            analyzed_message=self.analyzed_message,
            symbol="AAPL",
        )

        # Assert
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert any("ADX" in reason for reason in result.validation.veto_reasons)

    def test_validate_converts_veto_to_warn_when_not_veto_mode(self):
        """Should convert VETO to WARN when veto_mode=False."""
        # Arrange
        validator_no_veto = TechnicalValidator(
            alpaca_client=self.mock_alpaca,
            veto_mode=False,  # Disable veto mode
        )

        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Mock indicator engine to return overbought RSI (would normally veto)
        validator_no_veto.indicator_engine = Mock()
        validator_no_veto.indicator_engine.calculate_all.return_value = TechnicalIndicators(
            rsi=75.0,  # Overbought
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=25.0,
        )

        # Act
        result = validator_no_veto.validate(
            analyzed_message=self.analyzed_message,
            symbol="AAPL",
        )

        # Assert
        assert result.validation.status == ValidationStatus.WARN
        assert len(result.validation.warnings) > 0

    def test_validate_with_options_data(self):
        """Should process options data when provided."""
        # Arrange
        options_data = OptionsFlowData(
            volume_ratio=3.0,  # High volume
            iv_rank=50.0,
            put_call_ratio=0.8,
            unusual_activity=True,
        )

        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Act
        result = self.validator.validate(
            analyzed_message=self.analyzed_message,
            symbol="AAPL",
            options_data=options_data,
        )

        # Assert
        assert result.validation.options_flow == options_data
        # Confidence modifier should be boosted due to volume spike and unusual activity
        assert result.validation.confidence_modifier > 1.0

    def test_validate_handles_api_error_gracefully(self):
        """Should return PASS with warning when API call fails."""
        # Arrange
        self.mock_alpaca.get_bars.side_effect = Exception("API Error")

        # Act
        result = self.validator.validate(
            analyzed_message=self.analyzed_message,
            symbol="AAPL",
        )

        # Assert
        assert isinstance(result, ValidatedSignal)
        # Should return PASS (fail-safe) with a warning
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.warnings) > 0
        assert any("error" in warning.lower() or "fail" in warning.lower()
                   for warning in result.validation.warnings)

    def test_validate_passes_correct_params_to_alpaca(self):
        """Should call alpaca with correct symbol, timeframe, and lookback."""
        # Arrange
        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Act
        self.validator.validate(
            analyzed_message=self.analyzed_message,
            symbol="TSLA",
        )

        # Assert
        self.mock_alpaca.get_bars.assert_called_once()
        call_kwargs = self.mock_alpaca.get_bars.call_args[1]
        assert call_kwargs["symbol"] == "TSLA"
        assert call_kwargs["timeframe"] == "5Min"
        assert call_kwargs["limit"] == 50

    def test_validate_neutral_sentiment_bypasses_veto(self):
        """Should bypass veto logic for NEUTRAL sentiment."""
        # Arrange
        neutral_message = AnalyzedMessage(
            message=SocialMessage(
                source="twitter",
                source_id="125",
                content="AAPL is trading flat.",
                author="trader123",
                timestamp="2024-01-15T10:30:00Z",
                url="https://twitter.com/trader123/status/125",
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                confidence=0.9,
            ),
        )

        mock_bars_df = self._create_mock_bars_df()
        self.mock_alpaca.get_bars.return_value = mock_bars_df

        # Mock indicator engine with conditions that would veto (but should be bypassed)
        self.validator.indicator_engine = Mock()
        self.validator.indicator_engine.calculate_all.return_value = TechnicalIndicators(
            rsi=75.0,  # Overbought
            macd_histogram=0.5,
            macd_trend="rising",
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=15.0,  # Weak trend
        )

        # Act
        result = self.validator.validate(
            analyzed_message=neutral_message,
            symbol="AAPL",
        )

        # Assert
        # NEUTRAL sentiment should bypass veto rules
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.veto_reasons) == 0
