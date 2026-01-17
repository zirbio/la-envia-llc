# src/scoring/time_factors.py
"""Time-based factor calculations for scoring system."""
from datetime import datetime, time
from zoneinfo import ZoneInfo


class TimeFactorCalculator:
    """Calculates time-based penalties for signals.

    Market Hours (ET - Eastern Time):
    - Pre-market: 4:00 AM - 9:30 AM -> premarket_factor
    - Regular: 9:30 AM - 4:00 PM -> market_hours_factor (1.0)
    - After-hours: 4:00 PM - 8:00 PM -> afterhours_factor
    - Closed: 8:00 PM - 4:00 AM -> 0.5 (lowest factor)
    - Weekends: Saturday/Sunday -> 0.5 (lowest factor)

    Earnings Logic:
    - If symbol has earnings within earnings_proximity_days, apply earnings_factor

    Attributes:
        timezone: Timezone for market hours calculation.
        premarket_factor: Factor for pre-market hours (default 0.9).
        market_hours_factor: Factor for regular market hours (default 1.0).
        afterhours_factor: Factor for after-hours (default 0.8).
        earnings_factor: Factor when near earnings (default 0.7).
        earnings_proximity_days: Days before/after earnings to apply factor.
    """

    # Market hour boundaries (in ET)
    PREMARKET_START = time(4, 0)  # 4:00 AM
    MARKET_OPEN = time(9, 30)  # 9:30 AM
    MARKET_CLOSE = time(16, 0)  # 4:00 PM
    AFTERHOURS_END = time(20, 0)  # 8:00 PM

    # Lowest factor for closed market
    CLOSED_FACTOR = 0.5

    def __init__(
        self,
        timezone: str = "America/New_York",
        premarket_factor: float = 0.9,
        market_hours_factor: float = 1.0,
        afterhours_factor: float = 0.8,
        earnings_factor: float = 0.7,
        earnings_proximity_days: int = 3,
    ):
        """Initialize the time factor calculator.

        Args:
            timezone: Timezone for market hours (default America/New_York).
            premarket_factor: Factor for pre-market hours (default 0.9).
            market_hours_factor: Factor for regular market hours (default 1.0).
            afterhours_factor: Factor for after-hours (default 0.8).
            earnings_factor: Factor when near earnings (default 0.7).
            earnings_proximity_days: Days before/after earnings to apply factor.
        """
        self._timezone = ZoneInfo(timezone)
        self._premarket_factor = premarket_factor
        self._market_hours_factor = market_hours_factor
        self._afterhours_factor = afterhours_factor
        self._earnings_factor = earnings_factor
        self._earnings_proximity_days = earnings_proximity_days

    def _to_et(self, timestamp: datetime) -> datetime:
        """Convert timestamp to Eastern Time.

        Args:
            timestamp: Datetime to convert.

        Returns:
            Datetime in Eastern Time.
        """
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
        return timestamp.astimezone(self._timezone)

    def _is_weekend(self, timestamp: datetime) -> bool:
        """Check if timestamp is on a weekend.

        Args:
            timestamp: Datetime to check.

        Returns:
            True if Saturday (5) or Sunday (6).
        """
        et_time = self._to_et(timestamp)
        # weekday() returns 0=Monday, 5=Saturday, 6=Sunday
        return et_time.weekday() >= 5

    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours.

        Regular market hours: 9:30 AM - 4:00 PM ET

        Args:
            timestamp: Datetime to check.

        Returns:
            True if during regular market hours.
        """
        et_time = self._to_et(timestamp)
        current_time = et_time.time()
        return self.MARKET_OPEN <= current_time < self.MARKET_CLOSE

    def is_premarket(self, timestamp: datetime) -> bool:
        """Check if timestamp is during pre-market hours.

        Pre-market hours: 4:00 AM - 9:30 AM ET

        Args:
            timestamp: Datetime to check.

        Returns:
            True if during pre-market hours.
        """
        et_time = self._to_et(timestamp)
        current_time = et_time.time()
        return self.PREMARKET_START <= current_time < self.MARKET_OPEN

    def _is_afterhours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during after-hours.

        After-hours: 4:00 PM - 8:00 PM ET

        Args:
            timestamp: Datetime to check.

        Returns:
            True if during after-hours.
        """
        et_time = self._to_et(timestamp)
        current_time = et_time.time()
        return self.MARKET_CLOSE <= current_time < self.AFTERHOURS_END

    def _is_closed(self, timestamp: datetime) -> bool:
        """Check if timestamp is during closed market hours.

        Closed hours: 8:00 PM - 4:00 AM ET (overnight)

        Args:
            timestamp: Datetime to check.

        Returns:
            True if during closed hours.
        """
        et_time = self._to_et(timestamp)
        current_time = et_time.time()
        # Closed is either after 8 PM or before 4 AM
        return current_time >= self.AFTERHOURS_END or current_time < self.PREMARKET_START

    def _is_near_earnings(
        self,
        timestamp: datetime,
        symbol: str,
        earnings_dates: dict[str, datetime] | None,
    ) -> bool:
        """Check if symbol has earnings within proximity days.

        Args:
            timestamp: Current timestamp.
            symbol: Stock ticker symbol.
            earnings_dates: Dictionary mapping symbols to earnings dates.

        Returns:
            True if symbol has earnings within proximity days.
        """
        if earnings_dates is None or symbol not in earnings_dates:
            return False

        earnings_date = earnings_dates[symbol]

        # Make both timezone-aware for comparison
        if earnings_date.tzinfo is None:
            earnings_date = earnings_date.replace(tzinfo=self._timezone)

        # Use date only for comparison
        timestamp_date = self._to_et(timestamp).date()
        earnings_date_only = self._to_et(earnings_date).date()

        # Calculate days difference
        days_diff = abs((earnings_date_only - timestamp_date).days)

        return days_diff <= self._earnings_proximity_days

    def calculate_factor(
        self,
        timestamp: datetime,
        symbol: str,
        earnings_dates: dict[str, datetime] | None = None,
    ) -> tuple[float, list[str]]:
        """Calculate time factor and return reasons.

        The factor is calculated based on:
        1. Time of day (market hours, premarket, afterhours, closed)
        2. Day of week (weekends are closed)
        3. Earnings proximity (if within proximity days)

        Args:
            timestamp: Signal timestamp.
            symbol: Stock ticker symbol.
            earnings_dates: Optional dict mapping symbols to earnings dates.

        Returns:
            Tuple of (factor, list of reasons applied).
        """
        reasons: list[str] = []
        factor = 1.0

        # Check weekend first (takes precedence)
        if self._is_weekend(timestamp):
            reasons.append("weekend")
            factor = self.CLOSED_FACTOR
        elif self.is_market_hours(timestamp):
            reasons.append("market_hours")
            factor = self._market_hours_factor
        elif self.is_premarket(timestamp):
            reasons.append("premarket")
            factor = self._premarket_factor
        elif self._is_afterhours(timestamp):
            reasons.append("afterhours")
            factor = self._afterhours_factor
        else:
            # Closed hours (overnight)
            reasons.append("closed")
            factor = self.CLOSED_FACTOR

        # Apply earnings proximity penalty (multiplicative)
        if self._is_near_earnings(timestamp, symbol, earnings_dates):
            reasons.append("earnings_proximity")
            factor *= self._earnings_factor

        return factor, reasons
