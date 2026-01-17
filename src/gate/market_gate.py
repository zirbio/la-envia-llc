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
