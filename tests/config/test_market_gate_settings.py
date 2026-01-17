"""Tests for MarketGateSettings configuration."""

import pytest

from config.settings import Settings


class TestMarketGateSettings:
    """Tests for market_gate settings section."""

    def test_market_gate_defaults(self) -> None:
        """MarketGateSettings has correct defaults."""
        settings = Settings()
        gate = settings.market_gate

        assert gate.enabled is True
        assert gate.trading_start == "09:30"
        assert gate.trading_end == "16:00"
        assert gate.avoid_lunch is True
        assert gate.vix_max == 30.0
        assert gate.vix_elevated == 25.0
        assert gate.elevated_size_factor == 0.5
        assert gate.spy_min_volume == 500_000
        assert gate.qqq_min_volume == 300_000
        assert gate.choppy_detection_enabled is True
        assert gate.choppy_atr_ratio_threshold == 1.5
