"""Tests for MarketGate class."""

from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from gate.market_gate import MarketGate, MarketGateSettings
from gate.models import GateCheckResult


ET = ZoneInfo("America/New_York")


class TestMarketGateTradingHours:
    """Tests for trading hours check."""

    @pytest.fixture
    def settings(self) -> MarketGateSettings:
        """Default settings for tests."""
        return MarketGateSettings(
            enabled=True,
            trading_start="09:30",
            trading_end="16:00",
            avoid_lunch=True,
            lunch_start="11:30",
            lunch_end="14:00",
            spy_min_volume=500_000,
            qqq_min_volume=300_000,
            vix_max=30.0,
            vix_elevated=25.0,
            elevated_size_factor=0.5,
            choppy_detection_enabled=True,
            choppy_atr_ratio_threshold=1.5,
        )

    @pytest.fixture
    def mock_alpaca(self) -> MagicMock:
        """Mock Alpaca client."""
        return MagicMock()

    @pytest.fixture
    def mock_vix_fetcher(self) -> MagicMock:
        """Mock VIX fetcher."""
        fetcher = MagicMock()
        fetcher.fetch_vix.return_value = 18.0
        return fetcher

    def test_trading_hours_pass_during_market(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock, mock_vix_fetcher: MagicMock
    ) -> None:
        """Trading hours check passes during regular market hours."""
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # 10:30 AM ET - during market hours, not lunch
        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        result = gate._check_trading_hours(test_time)

        assert result.passed is True
        assert result.name == "trading_hours"

    def test_trading_hours_fail_before_open(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock, mock_vix_fetcher: MagicMock
    ) -> None:
        """Trading hours check fails before market open."""
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # 9:00 AM ET - before market open
        test_time = datetime(2026, 1, 17, 9, 0, tzinfo=ET)
        result = gate._check_trading_hours(test_time)

        assert result.passed is False
        assert "before market open" in result.reason.lower()

    def test_trading_hours_fail_after_close(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock, mock_vix_fetcher: MagicMock
    ) -> None:
        """Trading hours check fails after market close."""
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # 4:30 PM ET - after market close
        test_time = datetime(2026, 1, 17, 16, 30, tzinfo=ET)
        result = gate._check_trading_hours(test_time)

        assert result.passed is False
        assert "after market close" in result.reason.lower()

    def test_trading_hours_fail_during_lunch(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock, mock_vix_fetcher: MagicMock
    ) -> None:
        """Trading hours check fails during lunch when avoid_lunch enabled."""
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # 12:30 PM ET - during lunch
        test_time = datetime(2026, 1, 17, 12, 30, tzinfo=ET)
        result = gate._check_trading_hours(test_time)

        assert result.passed is False
        assert "lunch" in result.reason.lower()

    def test_trading_hours_pass_during_lunch_when_disabled(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock, mock_vix_fetcher: MagicMock
    ) -> None:
        """Trading hours check passes during lunch when avoid_lunch disabled."""
        settings.avoid_lunch = False
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # 12:30 PM ET - during lunch
        test_time = datetime(2026, 1, 17, 12, 30, tzinfo=ET)
        result = gate._check_trading_hours(test_time)

        assert result.passed is True


class TestMarketGateVolume:
    """Tests for volume check."""

    @pytest.fixture
    def settings(self) -> MarketGateSettings:
        """Default settings for tests."""
        return MarketGateSettings(
            spy_min_volume=500_000,
            qqq_min_volume=300_000,
        )

    @pytest.fixture
    def mock_vix_fetcher(self) -> MagicMock:
        """Mock VIX fetcher."""
        fetcher = MagicMock()
        fetcher.fetch_vix.return_value = 18.0
        return fetcher

    @pytest.mark.asyncio
    async def test_volume_pass_when_above_minimum(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Volume check passes when SPY and QQQ above minimum."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_volume()

        assert result.passed is True
        assert result.name == "volume"

    @pytest.mark.asyncio
    async def test_volume_fail_when_spy_low(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Volume check fails when SPY below minimum."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 100_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_volume()

        assert result.passed is False
        assert "SPY" in result.reason

    @pytest.mark.asyncio
    async def test_volume_fail_when_qqq_low(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Volume check fails when QQQ below minimum."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 100_000},
        ]

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_volume()

        assert result.passed is False
        assert "QQQ" in result.reason

    @pytest.mark.asyncio
    async def test_volume_handles_api_error(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Volume check fails gracefully on API error."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = Exception("API error")

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_volume()

        assert result.passed is False
        assert "error" in result.reason.lower()
