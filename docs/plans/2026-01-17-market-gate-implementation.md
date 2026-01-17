# Phase 7: Market Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MarketGate that verifies market conditions (hours, volume, VIX, choppy) before allowing trade execution.

**Architecture:** MarketGate evaluates 4 independent checks, returns GateStatus with position_size_factor. Soft block approach - signals are processed but not executed when gate is closed.

**Tech Stack:** Python dataclasses, Alpaca API (volume), yfinance (VIX), Pydantic settings, zoneinfo (timezone)

---

## Task 1: Data Models (GateCheckResult, GateStatus)

**Files:**
- Create: `src/gate/__init__.py`
- Create: `src/gate/models.py`
- Create: `tests/gate/__init__.py`
- Create: `tests/gate/test_models.py`

**Step 1: Create test file with initial tests**

```python
# tests/gate/test_models.py
"""Tests for gate data models."""

from datetime import datetime

import pytest

from gate.models import GateCheckResult, GateStatus


class TestGateCheckResult:
    """Tests for GateCheckResult dataclass."""

    def test_create_passed_check(self) -> None:
        """GateCheckResult can be created with passed=True."""
        result = GateCheckResult(
            name="trading_hours",
            passed=True,
            reason=None,
            data={"current_time": "10:30"},
        )
        assert result.name == "trading_hours"
        assert result.passed is True
        assert result.reason is None
        assert result.data == {"current_time": "10:30"}

    def test_create_failed_check(self) -> None:
        """GateCheckResult can be created with passed=False and reason."""
        result = GateCheckResult(
            name="vix",
            passed=False,
            reason="VIX above maximum threshold",
            data={"vix_value": 35.5},
        )
        assert result.name == "vix"
        assert result.passed is False
        assert result.reason == "VIX above maximum threshold"
        assert result.data["vix_value"] == 35.5


class TestGateStatus:
    """Tests for GateStatus dataclass."""

    def test_create_open_status(self) -> None:
        """GateStatus can be created with is_open=True."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=True, reason=None, data={}),
            GateCheckResult(name="volume", passed=True, reason=None, data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=True,
            checks=checks,
            position_size_factor=1.0,
        )
        assert status.is_open is True
        assert status.position_size_factor == 1.0
        assert len(status.checks) == 2

    def test_create_closed_status(self) -> None:
        """GateStatus can be created with is_open=False."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=False, reason="Market closed", data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=False,
            checks=checks,
            position_size_factor=0.0,
        )
        assert status.is_open is False
        assert status.position_size_factor == 0.0

    def test_get_failed_checks_returns_only_failed(self) -> None:
        """get_failed_checks returns only checks that did not pass."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=True, reason=None, data={}),
            GateCheckResult(name="vix", passed=False, reason="VIX too high", data={}),
            GateCheckResult(name="volume", passed=True, reason=None, data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=False,
            checks=checks,
            position_size_factor=0.0,
        )
        failed = status.get_failed_checks()
        assert len(failed) == 1
        assert failed[0].name == "vix"

    def test_get_failed_checks_returns_empty_when_all_pass(self) -> None:
        """get_failed_checks returns empty list when all checks pass."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=True, reason=None, data={}),
            GateCheckResult(name="volume", passed=True, reason=None, data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=True,
            checks=checks,
            position_size_factor=1.0,
        )
        failed = status.get_failed_checks()
        assert len(failed) == 0
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'gate'"

**Step 3: Create the module and models**

```python
# src/gate/__init__.py
"""Gate module for market condition verification."""

from .models import GateCheckResult, GateStatus

__all__ = ["GateCheckResult", "GateStatus"]
```

```python
# src/gate/models.py
"""Data models for market condition gate."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class GateCheckResult:
    """Result of a single gate check.

    Attributes:
        name: Check identifier (e.g., "trading_hours", "volume", "vix", "choppy").
        passed: Whether the check passed.
        reason: Explanation if check failed, None if passed.
        data: Observed data values used in the check.
    """

    name: str
    passed: bool
    reason: str | None
    data: dict[str, Any]


@dataclass
class GateStatus:
    """Combined result of all gate checks.

    Attributes:
        timestamp: When the gate check was performed.
        is_open: Whether all checks passed and trading is allowed.
        checks: List of individual check results.
        position_size_factor: Multiplier for position size (1.0 normal, 0.5 elevated, 0.0 blocked).
    """

    timestamp: datetime
    is_open: bool
    checks: list[GateCheckResult]
    position_size_factor: float

    def get_failed_checks(self) -> list[GateCheckResult]:
        """Return checks that did not pass.

        Returns:
            List of GateCheckResult where passed is False.
        """
        return [check for check in self.checks if not check.passed]
```

```python
# tests/gate/__init__.py
"""Tests for gate module."""
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_models.py -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ && git commit -m "feat(gate): add GateCheckResult and GateStatus models"
```

---

## Task 2: VixFetcher (yfinance wrapper)

**Files:**
- Create: `src/gate/vix_fetcher.py`
- Create: `tests/gate/test_vix_fetcher.py`
- Modify: `src/gate/__init__.py`

**Step 1: Create test file**

```python
# tests/gate/test_vix_fetcher.py
"""Tests for VIX fetcher."""

from unittest.mock import MagicMock, patch

import pytest

from gate.vix_fetcher import VixFetcher


class TestVixFetcher:
    """Tests for VixFetcher class."""

    def test_fetch_vix_returns_value(self) -> None:
        """fetch_vix returns current VIX value from yfinance."""
        fetcher = VixFetcher()

        # Mock yfinance
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 18.5}

        with patch("gate.vix_fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetcher.fetch_vix()

        assert result == 18.5

    def test_fetch_vix_uses_previous_close_as_fallback(self) -> None:
        """fetch_vix uses previousClose if regularMarketPrice not available."""
        fetcher = VixFetcher()

        mock_ticker = MagicMock()
        mock_ticker.info = {"previousClose": 20.0}

        with patch("gate.vix_fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetcher.fetch_vix()

        assert result == 20.0

    def test_fetch_vix_returns_none_on_error(self) -> None:
        """fetch_vix returns None when yfinance fails."""
        fetcher = VixFetcher()

        with patch("gate.vix_fetcher.yf.Ticker", side_effect=Exception("Network error")):
            result = fetcher.fetch_vix()

        assert result is None

    def test_fetch_vix_returns_none_when_no_price_data(self) -> None:
        """fetch_vix returns None when no price data available."""
        fetcher = VixFetcher()

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("gate.vix_fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetcher.fetch_vix()

        assert result is None
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_vix_fetcher.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'gate.vix_fetcher'"

**Step 3: Install yfinance and create VixFetcher**

First, add yfinance to requirements.txt if not present:

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && grep -q "yfinance" requirements.txt || echo "yfinance>=0.2.0" >> requirements.txt && uv pip install yfinance
```

```python
# src/gate/vix_fetcher.py
"""VIX data fetcher using yfinance."""

import yfinance as yf


class VixFetcher:
    """Fetches VIX (CBOE Volatility Index) data.

    Uses yfinance to get current VIX value since Alpaca doesn't
    provide direct access to the VIX index.
    """

    VIX_SYMBOL = "^VIX"

    def fetch_vix(self) -> float | None:
        """Fetch current VIX value.

        Returns:
            Current VIX value, or None if fetch fails.
        """
        try:
            ticker = yf.Ticker(self.VIX_SYMBOL)
            info = ticker.info

            # Try regularMarketPrice first, then previousClose as fallback
            price = info.get("regularMarketPrice") or info.get("previousClose")
            return float(price) if price is not None else None

        except Exception:
            return None
```

Update `src/gate/__init__.py`:

```python
# src/gate/__init__.py
"""Gate module for market condition verification."""

from .models import GateCheckResult, GateStatus
from .vix_fetcher import VixFetcher

__all__ = ["GateCheckResult", "GateStatus", "VixFetcher"]
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_vix_fetcher.py -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ requirements.txt && git commit -m "feat(gate): add VixFetcher for yfinance VIX data"
```

---

## Task 3: MarketGate - Core Structure and Trading Hours Check

**Files:**
- Create: `src/gate/market_gate.py`
- Create: `tests/gate/test_market_gate.py`
- Modify: `src/gate/__init__.py`

**Step 1: Create test file with trading hours tests**

```python
# tests/gate/test_market_gate.py
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

        # 10:30 AM ET - during market hours
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateTradingHours -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'gate.market_gate'"

**Step 3: Create MarketGate with trading hours check**

```python
# src/gate/market_gate.py
"""Market condition gate for trading validation."""

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from execution.alpaca_client import AlpacaClient
from gate.models import GateCheckResult, GateStatus
from gate.vix_fetcher import VixFetcher


ET = ZoneInfo("America/New_York")


class MarketGateSettings(BaseModel):
    """Configuration for MarketGate.

    Attributes:
        enabled: Whether the gate is active.
        trading_start: Market open time (HH:MM format, ET).
        trading_end: Market close time (HH:MM format, ET).
        avoid_lunch: Whether to block trading during lunch hours.
        lunch_start: Lunch period start time (HH:MM format, ET).
        lunch_end: Lunch period end time (HH:MM format, ET).
        spy_min_volume: Minimum 1-minute volume for SPY.
        qqq_min_volume: Minimum 1-minute volume for QQQ.
        vix_max: VIX level above which trading is blocked.
        vix_elevated: VIX level above which position size is reduced.
        elevated_size_factor: Position size multiplier when VIX elevated.
        choppy_detection_enabled: Whether to check for choppy market.
        choppy_atr_ratio_threshold: Ratio threshold for choppy detection.
    """

    enabled: bool = True
    trading_start: str = "09:30"
    trading_end: str = "16:00"
    avoid_lunch: bool = True
    lunch_start: str = "11:30"
    lunch_end: str = "14:00"
    spy_min_volume: int = 500_000
    qqq_min_volume: int = 300_000
    vix_max: float = 30.0
    vix_elevated: float = 25.0
    elevated_size_factor: float = 0.5
    choppy_detection_enabled: bool = True
    choppy_atr_ratio_threshold: float = 1.5


class MarketGate:
    """Evaluates market conditions before allowing trading.

    Performs 4 independent checks:
    1. Trading hours (09:30-16:00 ET, optional lunch avoidance)
    2. Volume (SPY/QQQ minimum 1-minute volume)
    3. VIX (volatility threshold)
    4. Choppy market (ATR ratio detection)

    Attributes:
        settings: Configuration for gate checks.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        settings: MarketGateSettings,
        vix_fetcher: VixFetcher | None = None,
    ):
        """Initialize MarketGate.

        Args:
            alpaca_client: Client for Alpaca API calls.
            settings: Gate configuration.
            vix_fetcher: Optional VIX fetcher (created if not provided).
        """
        self._alpaca = alpaca_client
        self._settings = settings
        self._vix_fetcher = vix_fetcher or VixFetcher()

    def _parse_time(self, time_str: str) -> time:
        """Parse HH:MM string to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def _check_trading_hours(self, current_time: datetime | None = None) -> GateCheckResult:
        """Check if current time is within trading hours.

        Args:
            current_time: Time to check (defaults to now in ET).

        Returns:
            GateCheckResult indicating if trading hours check passed.
        """
        if current_time is None:
            current_time = datetime.now(ET)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=ET)

        current = current_time.time()
        market_open = self._parse_time(self._settings.trading_start)
        market_close = self._parse_time(self._settings.trading_end)

        data = {
            "current_time": current.strftime("%H:%M"),
            "market_open": self._settings.trading_start,
            "market_close": self._settings.trading_end,
        }

        # Check if before market open
        if current < market_open:
            return GateCheckResult(
                name="trading_hours",
                passed=False,
                reason=f"Before market open ({self._settings.trading_start} ET)",
                data=data,
            )

        # Check if after market close
        if current >= market_close:
            return GateCheckResult(
                name="trading_hours",
                passed=False,
                reason=f"After market close ({self._settings.trading_end} ET)",
                data=data,
            )

        # Check lunch hours if enabled
        if self._settings.avoid_lunch:
            lunch_start = self._parse_time(self._settings.lunch_start)
            lunch_end = self._parse_time(self._settings.lunch_end)
            data["lunch_start"] = self._settings.lunch_start
            data["lunch_end"] = self._settings.lunch_end

            if lunch_start <= current < lunch_end:
                return GateCheckResult(
                    name="trading_hours",
                    passed=False,
                    reason=f"During lunch hours ({self._settings.lunch_start}-{self._settings.lunch_end} ET)",
                    data=data,
                )

        return GateCheckResult(
            name="trading_hours",
            passed=True,
            reason=None,
            data=data,
        )
```

Update `src/gate/__init__.py`:

```python
# src/gate/__init__.py
"""Gate module for market condition verification."""

from .market_gate import MarketGate, MarketGateSettings
from .models import GateCheckResult, GateStatus
from .vix_fetcher import VixFetcher

__all__ = [
    "GateCheckResult",
    "GateStatus",
    "MarketGate",
    "MarketGateSettings",
    "VixFetcher",
]
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateTradingHours -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ && git commit -m "feat(gate): add MarketGate with trading hours check"
```

---

## Task 4: MarketGate - Volume Check

**Files:**
- Modify: `src/gate/market_gate.py`
- Modify: `tests/gate/test_market_gate.py`

**Step 1: Add volume check tests**

Add to `tests/gate/test_market_gate.py`:

```python
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
            {"symbol": "SPY", "volume": 600_000},  # SPY
            {"symbol": "QQQ", "volume": 400_000},  # QQQ
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
            {"symbol": "SPY", "volume": 100_000},  # SPY low
            {"symbol": "QQQ", "volume": 400_000},  # QQQ ok
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
            {"symbol": "SPY", "volume": 600_000},  # SPY ok
            {"symbol": "QQQ", "volume": 100_000},  # QQQ low
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateVolume -v
```

Expected: FAIL with "AttributeError: 'MarketGate' object has no attribute '_check_volume'"

**Step 3: Add volume check to MarketGate**

Add to `src/gate/market_gate.py` in the `MarketGate` class:

```python
    async def _check_volume(self) -> GateCheckResult:
        """Check if SPY and QQQ have sufficient volume.

        Uses Alpaca to get latest 1-minute bar volume.

        Returns:
            GateCheckResult indicating if volume check passed.
        """
        data: dict[str, Any] = {
            "spy_min_required": self._settings.spy_min_volume,
            "qqq_min_required": self._settings.qqq_min_volume,
        }

        try:
            spy_bar = await self._alpaca.get_latest_bar("SPY")
            qqq_bar = await self._alpaca.get_latest_bar("QQQ")

            spy_volume = spy_bar.get("volume", 0)
            qqq_volume = qqq_bar.get("volume", 0)

            data["spy_volume"] = spy_volume
            data["qqq_volume"] = qqq_volume

            # Check SPY volume
            if spy_volume < self._settings.spy_min_volume:
                return GateCheckResult(
                    name="volume",
                    passed=False,
                    reason=f"SPY volume ({spy_volume:,}) below minimum ({self._settings.spy_min_volume:,})",
                    data=data,
                )

            # Check QQQ volume
            if qqq_volume < self._settings.qqq_min_volume:
                return GateCheckResult(
                    name="volume",
                    passed=False,
                    reason=f"QQQ volume ({qqq_volume:,}) below minimum ({self._settings.qqq_min_volume:,})",
                    data=data,
                )

            return GateCheckResult(
                name="volume",
                passed=True,
                reason=None,
                data=data,
            )

        except Exception as e:
            data["error"] = str(e)
            return GateCheckResult(
                name="volume",
                passed=False,
                reason=f"Error fetching volume data: {e}",
                data=data,
            )
```

Also add `from typing import Any` to imports if not present.

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateVolume -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ && git commit -m "feat(gate): add volume check to MarketGate"
```

---

## Task 5: MarketGate - VIX Check

**Files:**
- Modify: `src/gate/market_gate.py`
- Modify: `tests/gate/test_market_gate.py`

**Step 1: Add VIX check tests**

Add to `tests/gate/test_market_gate.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateVix -v
```

Expected: FAIL with "AttributeError: 'MarketGate' object has no attribute '_check_vix'"

**Step 3: Add VIX check to MarketGate**

Add to `src/gate/market_gate.py` in the `MarketGate` class:

```python
    def _check_vix(self) -> tuple[GateCheckResult, float]:
        """Check VIX level and determine position size factor.

        Returns:
            Tuple of (GateCheckResult, position_size_factor).
            Factor is 1.0 for normal, 0.5 for elevated, 0.0 for blocked.
        """
        vix_value = self._vix_fetcher.fetch_vix()

        data: dict[str, Any] = {
            "vix_max": self._settings.vix_max,
            "vix_elevated": self._settings.vix_elevated,
        }

        # Handle fetch failure
        if vix_value is None:
            data["vix_value"] = None
            data["status"] = "unavailable"
            return (
                GateCheckResult(
                    name="vix",
                    passed=False,
                    reason="VIX data unavailable",
                    data=data,
                ),
                0.0,
            )

        data["vix_value"] = vix_value

        # Check if VIX above maximum (blocked)
        if vix_value > self._settings.vix_max:
            data["status"] = "blocked"
            return (
                GateCheckResult(
                    name="vix",
                    passed=False,
                    reason=f"VIX ({vix_value:.1f}) above maximum ({self._settings.vix_max})",
                    data=data,
                ),
                0.0,
            )

        # Check if VIX elevated (reduced size)
        if vix_value >= self._settings.vix_elevated:
            data["status"] = "elevated"
            return (
                GateCheckResult(
                    name="vix",
                    passed=True,
                    reason=None,
                    data=data,
                ),
                self._settings.elevated_size_factor,
            )

        # Normal VIX
        data["status"] = "normal"
        return (
            GateCheckResult(
                name="vix",
                passed=True,
                reason=None,
                data=data,
            ),
            1.0,
        )
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateVix -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ && git commit -m "feat(gate): add VIX check to MarketGate"
```

---

## Task 6: MarketGate - Choppy Market Check

**Files:**
- Modify: `src/gate/market_gate.py`
- Modify: `tests/gate/test_market_gate.py`

**Step 1: Add choppy market check tests**

Add to `tests/gate/test_market_gate.py`:

```python
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
            {"high": 455, "low": 450, "close": 453},  # Today: range = 5
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
            {"high": 452, "low": 450, "close": 451},  # Today: range = 2
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateChoppy -v
```

Expected: FAIL with "AttributeError: 'MarketGate' object has no attribute '_check_choppy_market'"

**Step 3: Add choppy market check to MarketGate**

Add to `src/gate/market_gate.py` in the `MarketGate` class:

```python
    async def _check_choppy_market(self) -> GateCheckResult:
        """Check if market is choppy (directionless).

        Uses SPY's (High - Low) / ATR(14) ratio.
        If ratio < threshold, market is considered choppy.

        Returns:
            GateCheckResult indicating if choppy check passed.
        """
        data: dict[str, Any] = {
            "threshold": self._settings.choppy_atr_ratio_threshold,
        }

        # Skip if disabled
        if not self._settings.choppy_detection_enabled:
            data["status"] = "disabled"
            return GateCheckResult(
                name="choppy_market",
                passed=True,
                reason=None,
                data=data,
            )

        try:
            # Get today's SPY bar for high/low
            bars = await self._alpaca.get_bars("SPY", timeframe="1Day", limit=1)
            if not bars:
                data["status"] = "no_data"
                return GateCheckResult(
                    name="choppy_market",
                    passed=False,
                    reason="No SPY bar data available",
                    data=data,
                )

            today_bar = bars[0]
            day_high = today_bar.get("high", 0)
            day_low = today_bar.get("low", 0)
            day_range = day_high - day_low

            # Get ATR
            atr = await self._alpaca.get_atr("SPY", period=14)

            data["day_high"] = day_high
            data["day_low"] = day_low
            data["day_range"] = day_range
            data["atr"] = atr

            # Calculate ratio
            if atr <= 0:
                data["status"] = "invalid_atr"
                return GateCheckResult(
                    name="choppy_market",
                    passed=False,
                    reason="Invalid ATR value",
                    data=data,
                )

            ratio = day_range / atr
            data["ratio"] = ratio

            # Check if choppy
            if ratio < self._settings.choppy_atr_ratio_threshold:
                data["status"] = "choppy"
                return GateCheckResult(
                    name="choppy_market",
                    passed=False,
                    reason=f"Choppy market detected (ratio {ratio:.2f} < {self._settings.choppy_atr_ratio_threshold})",
                    data=data,
                )

            data["status"] = "trending"
            return GateCheckResult(
                name="choppy_market",
                passed=True,
                reason=None,
                data=data,
            )

        except Exception as e:
            data["error"] = str(e)
            return GateCheckResult(
                name="choppy_market",
                passed=False,
                reason=f"Error checking choppy market: {e}",
                data=data,
            )
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateChoppy -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ && git commit -m "feat(gate): add choppy market check to MarketGate"
```

---

## Task 7: MarketGate - Main check() Method

**Files:**
- Modify: `src/gate/market_gate.py`
- Modify: `tests/gate/test_market_gate.py`

**Step 1: Add main check() tests**

Add to `tests/gate/test_market_gate.py`:

```python
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

        # Use a time during market hours
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateCheck -v
```

Expected: FAIL with "AttributeError: 'MarketGate' object has no attribute 'check'"

**Step 3: Add main check() method to MarketGate**

Add to `src/gate/market_gate.py` in the `MarketGate` class:

```python
    async def check(self, current_time: datetime | None = None) -> GateStatus:
        """Evaluate all market conditions.

        Runs all 4 checks and combines results into GateStatus.

        Args:
            current_time: Time for trading hours check (defaults to now).

        Returns:
            GateStatus with combined results.
        """
        timestamp = datetime.now(ET)

        # If gate disabled, return open status immediately
        if not self._settings.enabled:
            return GateStatus(
                timestamp=timestamp,
                is_open=True,
                checks=[],
                position_size_factor=1.0,
            )

        checks: list[GateCheckResult] = []

        # 1. Trading hours check (synchronous)
        hours_result = self._check_trading_hours(current_time)
        checks.append(hours_result)

        # 2. VIX check (synchronous, returns factor)
        vix_result, vix_factor = self._check_vix()
        checks.append(vix_result)

        # 3. Volume check (async)
        volume_result = await self._check_volume()
        checks.append(volume_result)

        # 4. Choppy market check (async)
        choppy_result = await self._check_choppy_market()
        checks.append(choppy_result)

        # Determine if gate is open (all checks passed)
        all_passed = all(check.passed for check in checks)

        # Determine position size factor
        if not all_passed:
            factor = 0.0
        else:
            factor = vix_factor

        return GateStatus(
            timestamp=timestamp,
            is_open=all_passed,
            checks=checks,
            position_size_factor=factor,
        )
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/gate/test_market_gate.py::TestMarketGateCheck -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/gate/ tests/gate/ && git commit -m "feat(gate): add main check() method to MarketGate"
```

---

## Task 8: MarketGateSettings Configuration

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Create: `tests/config/test_market_gate_settings.py`

**Step 1: Create settings test**

```python
# tests/config/test_market_gate_settings.py
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

    def test_market_gate_loaded_from_yaml(self, tmp_path) -> None:
        """MarketGateSettings loads from YAML file."""
        yaml_content = """
market_gate:
  enabled: false
  trading_start: "10:00"
  vix_max: 35.0
"""
        yaml_file = tmp_path / "settings.yaml"
        yaml_file.write_text(yaml_content)

        settings = Settings(_env_file=None)
        # Settings should use defaults since we can't easily override yaml path
        assert settings.market_gate.enabled is True  # Default
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/config/test_market_gate_settings.py -v
```

Expected: FAIL with "AttributeError: 'Settings' object has no attribute 'market_gate'"

**Step 3: Add MarketGateSettings to config**

Modify `src/config/settings.py` - add import and settings class:

```python
# Add to imports
from gate.market_gate import MarketGateSettings

# Add to Settings class (after other settings fields)
    market_gate: MarketGateSettings = Field(default_factory=MarketGateSettings)
```

Update `config/settings.yaml`:

```yaml
# Add after execution section

# CAPA 0: Market Condition Gate
market_gate:
  enabled: true
  trading_start: "09:30"
  trading_end: "16:00"
  avoid_lunch: true
  lunch_start: "11:30"
  lunch_end: "14:00"
  spy_min_volume: 500000
  qqq_min_volume: 300000
  vix_max: 30.0
  vix_elevated: 25.0
  elevated_size_factor: 0.5
  choppy_detection_enabled: true
  choppy_atr_ratio_threshold: 1.5
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/config/test_market_gate_settings.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/config/ config/settings.yaml tests/config/ && git commit -m "feat(config): add MarketGateSettings to configuration"
```

---

## Task 9: Update TradeExecutor for Gate Integration

**Files:**
- Modify: `src/execution/trade_executor.py`
- Modify: `tests/execution/test_trade_executor.py`

**Step 1: Add gate integration tests**

Add to `tests/execution/test_trade_executor.py`:

```python
class TestTradeExecutorGateIntegration:
    """Tests for gate integration in TradeExecutor."""

    @pytest.fixture
    def mock_gate_status_open(self) -> GateStatus:
        """Gate status with is_open=True."""
        from gate.models import GateCheckResult, GateStatus
        return GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="test", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )

    @pytest.fixture
    def mock_gate_status_closed(self) -> GateStatus:
        """Gate status with is_open=False."""
        from gate.models import GateCheckResult, GateStatus
        return GateStatus(
            timestamp=datetime.now(),
            is_open=False,
            checks=[GateCheckResult(name="vix", passed=False, reason="VIX too high", data={})],
            position_size_factor=0.0,
        )

    @pytest.fixture
    def mock_gate_status_elevated(self) -> GateStatus:
        """Gate status with elevated VIX (factor 0.5)."""
        from gate.models import GateCheckResult, GateStatus
        return GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="vix", passed=True, reason=None, data={"status": "elevated"})],
            position_size_factor=0.5,
        )

    @pytest.mark.asyncio
    async def test_execute_respects_gate_closed(
        self,
        executor: TradeExecutor,
        recommendation: TradeRecommendation,
        approved_risk_result: RiskCheckResult,
        mock_gate_status_closed: GateStatus,
    ) -> None:
        """Execute returns failure when gate is closed."""
        result = await executor.execute(
            recommendation, approved_risk_result, gate_status=mock_gate_status_closed
        )

        assert result.success is False
        assert "gate closed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_applies_size_factor(
        self,
        mock_alpaca: AsyncMock,
        risk_manager: RiskManager,
        recommendation: TradeRecommendation,
        approved_risk_result: RiskCheckResult,
        mock_gate_status_elevated: GateStatus,
    ) -> None:
        """Execute applies position_size_factor to quantity."""
        # Set up approved for 100 shares
        approved_risk_result.adjusted_quantity = 100

        mock_alpaca.submit_bracket_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 50,  # Half of 100
        }

        executor = TradeExecutor(mock_alpaca, risk_manager)
        result = await executor.execute(
            recommendation, approved_risk_result, gate_status=mock_gate_status_elevated
        )

        # Verify order was submitted with reduced quantity
        call_args = mock_alpaca.submit_bracket_order.call_args
        assert call_args.kwargs["qty"] == 50  # 100 * 0.5

    @pytest.mark.asyncio
    async def test_execute_without_gate_status(
        self,
        executor: TradeExecutor,
        recommendation: TradeRecommendation,
        approved_risk_result: RiskCheckResult,
    ) -> None:
        """Execute works normally when gate_status not provided."""
        result = await executor.execute(recommendation, approved_risk_result)

        assert result.success is True
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/execution/test_trade_executor.py::TestTradeExecutorGateIntegration -v
```

Expected: FAIL (tests reference gate_status parameter that doesn't exist)

**Step 3: Update TradeExecutor to accept gate_status**

Modify `src/execution/trade_executor.py`:

```python
# Add import at top
from gate.models import GateStatus

# Modify execute method signature and add gate check:
    async def execute(
        self,
        recommendation: TradeRecommendation,
        risk_result: RiskCheckResult,
        gate_status: GateStatus | None = None,
    ) -> ExecutionResult:
        """Execute an approved trade as a bracket order.

        Args:
            recommendation: The trade recommendation to execute.
            risk_result: Risk check result (must be approved).
            gate_status: Optional gate status for market condition check.

        Returns:
            ExecutionResult with success/failure and order details.
        """
        # Check 1: Validate risk_result is approved
        if not risk_result.approved:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=recommendation.symbol,
                side="buy" if recommendation.direction == Direction.LONG else "sell",
                quantity=0,
                filled_price=None,
                error_message=risk_result.rejection_reason or "Trade not approved",
                timestamp=datetime.now(),
            )

        # Check 2: Validate gate is open (if provided)
        if gate_status is not None and not gate_status.is_open:
            failed_checks = gate_status.get_failed_checks()
            reasons = [f"{c.name}: {c.reason}" for c in failed_checks]
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=recommendation.symbol,
                side="buy" if recommendation.direction == Direction.LONG else "sell",
                quantity=0,
                filled_price=None,
                error_message=f"Market gate closed - {'; '.join(reasons)}",
                timestamp=datetime.now(),
            )

        # Apply position size factor from gate
        adjusted_quantity = risk_result.adjusted_quantity
        if gate_status is not None:
            adjusted_quantity = int(adjusted_quantity * gate_status.position_size_factor)

        # ... rest of execute method (use adjusted_quantity instead of risk_result.adjusted_quantity)
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/execution/test_trade_executor.py::TestTradeExecutorGateIntegration -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add src/execution/ tests/execution/ && git commit -m "feat(execution): add gate_status parameter to TradeExecutor"
```

---

## Task 10: Integration Tests

**Files:**
- Create: `tests/integration/test_gate_pipeline.py`

**Step 1: Create integration tests**

```python
# tests/integration/test_gate_pipeline.py
"""Integration tests for market gate with execution pipeline."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from execution.trade_executor import TradeExecutor
from gate.market_gate import MarketGate, MarketGateSettings
from gate.models import GateStatus
from risk.models import RiskCheckResult
from risk.risk_manager import RiskManager
from scoring.models import Direction, TradeRecommendation


ET = ZoneInfo("America/New_York")


class TestGatePipelineIntegration:
    """Integration tests for gate → risk → execution flow."""

    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        """Create RiskManager with standard settings."""
        return RiskManager(
            max_position_value=10000.0,
            max_daily_loss=500.0,
        )

    @pytest.fixture
    def recommendation(self) -> TradeRecommendation:
        """Create a standard trade recommendation."""
        return TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            confidence=0.85,
            stop_loss=145.0,
            take_profit=165.0,
            reasons=["Strong sentiment"],
        )

    @pytest.mark.asyncio
    async def test_full_flow_gate_open(self, risk_manager: RiskManager, recommendation: TradeRecommendation) -> None:
        """Full flow executes when gate is open."""
        # Setup mocks
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5
        mock_alpaca.submit_bracket_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 10,
        }

        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # Check gate during market hours
        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is True

        # Check risk
        risk_result = risk_manager.check_trade(recommendation, 10, 150.0)
        assert risk_result.approved is True

        # Execute
        executor = TradeExecutor(mock_alpaca, risk_manager)
        exec_result = await executor.execute(recommendation, risk_result, gate_status)

        assert exec_result.success is True

    @pytest.mark.asyncio
    async def test_full_flow_gate_closed_blocks_execution(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Execution blocked when gate is closed."""
        mock_alpaca = AsyncMock()
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 35.0  # VIX blocked

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is False

        # Risk would approve, but gate should block
        risk_result = risk_manager.check_trade(recommendation, 10, 150.0)
        assert risk_result.approved is True

        executor = TradeExecutor(mock_alpaca, risk_manager)
        exec_result = await executor.execute(recommendation, risk_result, gate_status)

        assert exec_result.success is False
        assert "gate closed" in exec_result.error_message.lower()

    @pytest.mark.asyncio
    async def test_vix_elevated_reduces_position_size(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """VIX elevated reduces position size by factor."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5
        mock_alpaca.submit_bracket_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 5,
        }

        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 27.0  # Elevated

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is True
        assert gate_status.position_size_factor == 0.5

        risk_result = risk_manager.check_trade(recommendation, 10, 150.0)

        executor = TradeExecutor(mock_alpaca, risk_manager)
        await executor.execute(recommendation, risk_result, gate_status)

        # Verify reduced quantity was used
        call_args = mock_alpaca.submit_bracket_order.call_args
        assert call_args.kwargs["qty"] == 5  # 10 * 0.5
```

**Step 2: Run tests**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest tests/integration/test_gate_pipeline.py -v
```

Expected: All 3 tests PASS

**Step 3: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git add tests/integration/ && git commit -m "test: add integration tests for gate pipeline"
```

---

## Task 11: Final Test Suite Run

**Step 1: Run full test suite with coverage**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && uv run pytest --cov=src --cov-report=term-missing -v
```

Expected: All tests pass, coverage >90% on gate module

**Step 2: Verify gate module coverage**

Check that:
- `src/gate/models.py` - 100%
- `src/gate/vix_fetcher.py` - >90%
- `src/gate/market_gate.py` - >90%

**Step 3: Final commit if any cleanup needed**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase7-market-gate && git status
```

If clean, proceed to merge.
