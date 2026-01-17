"""Market condition gate for trading validation."""

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from execution.alpaca_client import AlpacaClient
from gate.models import GateCheckResult, GateStatus
from gate.vix_fetcher import VixFetcher


ET = ZoneInfo("America/New_York")


class MarketGateSettings(BaseModel):
    """Configuration for MarketGate."""

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
    """Evaluates market conditions before allowing trading."""

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        settings: MarketGateSettings,
        vix_fetcher: VixFetcher | None = None,
    ):
        self._alpaca = alpaca_client
        self._settings = settings
        self._vix_fetcher = vix_fetcher or VixFetcher()

    def _parse_time(self, time_str: str) -> time:
        """Parse HH:MM string to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def _check_trading_hours(self, current_time: datetime | None = None) -> GateCheckResult:
        """Check if current time is within trading hours."""
        if current_time is None:
            current_time = datetime.now(ET)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=ET)

        current = current_time.time()
        market_open = self._parse_time(self._settings.trading_start)
        market_close = self._parse_time(self._settings.trading_end)

        data: dict[str, Any] = {
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
