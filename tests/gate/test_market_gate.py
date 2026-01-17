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


class TestMarketGateVix:
    """Tests for VIX check."""

    @pytest.fixture
    def settings(self) -> MarketGateSettings:
        """Default settings for tests."""
        return MarketGateSettings(
            vix_max=30.0,
            vix_elevated=25.0,
            elevated_size_factor=0.5,
        )

    @pytest.fixture
    def mock_alpaca(self) -> MagicMock:
        """Mock Alpaca client."""
        return MagicMock()

    def test_vix_normal_returns_factor_one(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock
    ) -> None:
        """VIX below elevated threshold returns factor 1.0."""
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result, factor = gate._check_vix()

        assert result.passed is True
        assert factor == 1.0
        assert result.data["vix_value"] == 18.0

    def test_vix_elevated_returns_reduced_factor(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock
    ) -> None:
        """VIX between elevated and max returns reduced factor."""
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 27.0

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result, factor = gate._check_vix()

        assert result.passed is True
        assert factor == 0.5
        assert "elevated" in result.data.get("status", "").lower()

    def test_vix_blocked_when_above_max(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock
    ) -> None:
        """VIX above max threshold blocks trading."""
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 35.0

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result, factor = gate._check_vix()

        assert result.passed is False
        assert factor == 0.0
        assert "above maximum" in result.reason.lower()

    def test_vix_handles_fetch_error(
        self, settings: MarketGateSettings, mock_alpaca: MagicMock
    ) -> None:
        """VIX check fails gracefully when fetch returns None."""
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = None

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result, factor = gate._check_vix()

        assert result.passed is False
        assert factor == 0.0
        assert "unavailable" in result.reason.lower()


class TestMarketGateChoppy:
    """Tests for choppy market check."""

    @pytest.fixture
    def settings(self) -> MarketGateSettings:
        """Default settings for tests."""
        return MarketGateSettings(
            choppy_detection_enabled=True,
            choppy_atr_ratio_threshold=1.5,
        )

    @pytest.fixture
    def mock_vix_fetcher(self) -> MagicMock:
        """Mock VIX fetcher."""
        fetcher = MagicMock()
        fetcher.fetch_vix.return_value = 18.0
        return fetcher

    @pytest.mark.asyncio
    async def test_choppy_pass_when_ratio_above_threshold(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Choppy check passes when range/ATR ratio above threshold."""
        mock_alpaca = AsyncMock()
        # Day range = 5, ATR = 2.5, ratio = 2.0 (above 1.5 threshold)
        mock_alpaca.get_bars.return_value = [
            {"high": 455, "low": 450, "close": 453},
        ]
        mock_alpaca.get_atr.return_value = 2.5

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_choppy_market()

        assert result.passed is True
        assert result.name == "choppy_market"

    @pytest.mark.asyncio
    async def test_choppy_fail_when_ratio_below_threshold(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Choppy check fails when range/ATR ratio below threshold."""
        mock_alpaca = AsyncMock()
        # Day range = 2, ATR = 2.5, ratio = 0.8 (below 1.5 threshold)
        mock_alpaca.get_bars.return_value = [
            {"high": 452, "low": 450, "close": 451},
        ]
        mock_alpaca.get_atr.return_value = 2.5

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_choppy_market()

        assert result.passed is False
        assert "choppy" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_choppy_skip_when_disabled(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Choppy check passes automatically when disabled."""
        settings.choppy_detection_enabled = False
        mock_alpaca = AsyncMock()

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_choppy_market()

        assert result.passed is True
        assert "disabled" in result.data.get("status", "").lower()

    @pytest.mark.asyncio
    async def test_choppy_handles_api_error(
        self, settings: MarketGateSettings, mock_vix_fetcher: MagicMock
    ) -> None:
        """Choppy check fails gracefully on API error."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_bars.side_effect = Exception("API error")

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        result = await gate._check_choppy_market()

        assert result.passed is False
        assert "error" in result.reason.lower()


class TestMarketGateCheck:
    """Tests for main check() method."""

    @pytest.fixture
    def settings(self) -> MarketGateSettings:
        """Default settings for tests."""
        return MarketGateSettings()

    @pytest.mark.asyncio
    async def test_check_returns_open_when_all_pass(self, settings: MarketGateSettings) -> None:
        """check() returns is_open=True when all checks pass."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5

        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # Use a time during market hours (not lunch)
        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        status = await gate.check(current_time=test_time)

        assert status.is_open is True
        assert status.position_size_factor == 1.0
        assert len(status.checks) == 4

    @pytest.mark.asyncio
    async def test_check_returns_closed_when_any_fail(self, settings: MarketGateSettings) -> None:
        """check() returns is_open=False when any check fails."""
        mock_alpaca = AsyncMock()
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 35.0  # VIX blocked

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # Even during market hours, VIX blocks
        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        status = await gate.check(current_time=test_time)

        assert status.is_open is False
        assert status.position_size_factor == 0.0

    @pytest.mark.asyncio
    async def test_check_returns_reduced_factor_when_vix_elevated(
        self, settings: MarketGateSettings
    ) -> None:
        """check() returns factor 0.5 when VIX elevated but not blocked."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5

        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 27.0  # Elevated

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        status = await gate.check(current_time=test_time)

        assert status.is_open is True
        assert status.position_size_factor == 0.5

    @pytest.mark.asyncio
    async def test_check_disabled_returns_open(self, settings: MarketGateSettings) -> None:
        """check() returns is_open=True with factor 1.0 when gate disabled."""
        settings.enabled = False
        mock_alpaca = AsyncMock()
        mock_vix_fetcher = MagicMock()

        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)
        status = await gate.check()

        assert status.is_open is True
        assert status.position_size_factor == 1.0
        assert len(status.checks) == 0
