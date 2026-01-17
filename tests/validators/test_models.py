# tests/validators/test_models.py
"""Tests for technical validation data models."""

from datetime import datetime

import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentLabel, SentimentResult
from src.models.social_message import SocialMessage
from src.validators.models import (
    OptionsFlowData,
    TechnicalIndicators,
    TechnicalValidation,
    ValidatedSignal,
    ValidationStatus,
)


class TestValidationStatus:
    """Test ValidationStatus enum."""

    def test_validation_status_has_pass(self):
        """ValidationStatus should have PASS value."""
        assert ValidationStatus.PASS is not None

    def test_validation_status_has_veto(self):
        """ValidationStatus should have VETO value."""
        assert ValidationStatus.VETO is not None

    def test_validation_status_has_warn(self):
        """ValidationStatus should have WARN value."""
        assert ValidationStatus.WARN is not None

    def test_validation_status_distinct_values(self):
        """ValidationStatus values should be distinct."""
        assert ValidationStatus.PASS != ValidationStatus.VETO
        assert ValidationStatus.PASS != ValidationStatus.WARN
        assert ValidationStatus.VETO != ValidationStatus.WARN


class TestTechnicalIndicators:
    """Test TechnicalIndicators dataclass."""

    def test_technical_indicators_creation(self):
        """Should create TechnicalIndicators with all required fields."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )
        assert indicators.rsi == 65.5
        assert indicators.macd_histogram == 0.25
        assert indicators.macd_trend == "rising"
        assert indicators.stochastic_k == 70.0
        assert indicators.stochastic_d == 68.0
        assert indicators.adx == 25.5

    def test_technical_indicators_optional_fields(self):
        """Should handle optional divergence fields."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
            rsi_divergence="bullish",
            macd_divergence="bearish",
        )
        assert indicators.rsi_divergence == "bullish"
        assert indicators.macd_divergence == "bearish"

    def test_technical_indicators_optional_defaults_to_none(self):
        """Optional fields should default to None."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )
        assert indicators.rsi_divergence is None
        assert indicators.macd_divergence is None

    def test_technical_indicators_macd_trend_values(self):
        """Should accept valid MACD trend values."""
        for trend in ["rising", "falling", "flat"]:
            indicators = TechnicalIndicators(
                rsi=50.0,
                macd_histogram=0.0,
                macd_trend=trend,
                stochastic_k=50.0,
                stochastic_d=50.0,
                adx=20.0,
            )
            assert indicators.macd_trend == trend


class TestOptionsFlowData:
    """Test OptionsFlowData dataclass."""

    def test_options_flow_data_creation(self):
        """Should create OptionsFlowData with all required fields."""
        flow = OptionsFlowData(
            volume_ratio=2.5,
            iv_rank=75.0,
            put_call_ratio=0.85,
            unusual_activity=True,
        )
        assert flow.volume_ratio == 2.5
        assert flow.iv_rank == 75.0
        assert flow.put_call_ratio == 0.85
        assert flow.unusual_activity is True

    def test_options_flow_data_with_false_unusual_activity(self):
        """Should handle unusual_activity=False."""
        flow = OptionsFlowData(
            volume_ratio=1.0,
            iv_rank=50.0,
            put_call_ratio=1.0,
            unusual_activity=False,
        )
        assert flow.unusual_activity is False


class TestTechnicalValidation:
    """Test TechnicalValidation dataclass."""

    def test_technical_validation_creation(self):
        """Should create TechnicalValidation with all required fields."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )

        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=indicators,
            confidence_modifier=1.0,
        )

        assert validation.status == ValidationStatus.PASS
        assert validation.indicators == indicators
        assert validation.confidence_modifier == 1.0

    def test_technical_validation_with_options_flow(self):
        """Should handle optional options_flow field."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )

        flow = OptionsFlowData(
            volume_ratio=2.5,
            iv_rank=75.0,
            put_call_ratio=0.85,
            unusual_activity=True,
        )

        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=indicators,
            options_flow=flow,
            confidence_modifier=1.2,
        )

        assert validation.options_flow == flow

    def test_technical_validation_default_empty_lists(self):
        """Should default veto_reasons and warnings to empty lists."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )

        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=indicators,
            confidence_modifier=1.0,
        )

        assert validation.veto_reasons == []
        assert validation.warnings == []

    def test_technical_validation_with_veto_reasons(self):
        """Should handle veto_reasons list."""
        indicators = TechnicalIndicators(
            rsi=85.0,
            macd_histogram=-0.25,
            macd_trend="falling",
            stochastic_k=90.0,
            stochastic_d=88.0,
            adx=15.0,
        )

        validation = TechnicalValidation(
            status=ValidationStatus.VETO,
            indicators=indicators,
            veto_reasons=["RSI overbought", "Weak trend (ADX < 20)"],
            confidence_modifier=0.0,
        )

        assert validation.veto_reasons == ["RSI overbought", "Weak trend (ADX < 20)"]

    def test_technical_validation_with_warnings(self):
        """Should handle warnings list."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )

        validation = TechnicalValidation(
            status=ValidationStatus.WARN,
            indicators=indicators,
            warnings=["Approaching overbought territory"],
            confidence_modifier=0.8,
        )

        assert validation.warnings == ["Approaching overbought territory"]


class TestValidatedSignal:
    """Test ValidatedSignal dataclass."""

    @pytest.fixture
    def sample_analyzed_message(self):
        """Create a sample AnalyzedMessage for testing."""
        social_msg = SocialMessage(
            source="twitter",
            source_id="123",
            content="$TSLA looking bullish",
            author="trader123",
            timestamp=datetime.now(),
            metadata={},
        )

        sentiment = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.95,
            confidence=0.9,
        )

        return AnalyzedMessage(
            message=social_msg,
            sentiment=sentiment,
        )

    @pytest.fixture
    def sample_validation_pass(self):
        """Create a sample PASS validation."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )

        return TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=indicators,
            confidence_modifier=1.0,
        )

    @pytest.fixture
    def sample_validation_veto(self):
        """Create a sample VETO validation."""
        indicators = TechnicalIndicators(
            rsi=85.0,
            macd_histogram=-0.25,
            macd_trend="falling",
            stochastic_k=90.0,
            stochastic_d=88.0,
            adx=15.0,
        )

        return TechnicalValidation(
            status=ValidationStatus.VETO,
            indicators=indicators,
            veto_reasons=["RSI overbought"],
            confidence_modifier=0.0,
        )

    def test_validated_signal_creation(
        self, sample_analyzed_message, sample_validation_pass
    ):
        """Should create ValidatedSignal with message and validation."""
        signal = ValidatedSignal(
            message=sample_analyzed_message,
            validation=sample_validation_pass,
        )

        assert signal.message == sample_analyzed_message
        assert signal.validation == sample_validation_pass

    def test_should_trade_returns_true_when_not_veto(
        self, sample_analyzed_message, sample_validation_pass
    ):
        """should_trade() should return True when status is not VETO."""
        signal = ValidatedSignal(
            message=sample_analyzed_message,
            validation=sample_validation_pass,
        )

        assert signal.should_trade() is True

    def test_should_trade_returns_false_when_veto(
        self, sample_analyzed_message, sample_validation_veto
    ):
        """should_trade() should return False when status is VETO."""
        signal = ValidatedSignal(
            message=sample_analyzed_message,
            validation=sample_validation_veto,
        )

        assert signal.should_trade() is False

    def test_should_trade_returns_true_for_warn_status(self, sample_analyzed_message):
        """should_trade() should return True for WARN status."""
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=70.0,
            stochastic_d=68.0,
            adx=25.5,
        )

        validation = TechnicalValidation(
            status=ValidationStatus.WARN,
            indicators=indicators,
            warnings=["Minor concern"],
            confidence_modifier=0.8,
        )

        signal = ValidatedSignal(
            message=sample_analyzed_message,
            validation=validation,
        )

        assert signal.should_trade() is True
