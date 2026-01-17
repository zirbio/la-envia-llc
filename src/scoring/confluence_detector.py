# src/scoring/confluence_detector.py
"""Confluence detector for tracking multiple signals on same ticker."""
from collections import defaultdict
from datetime import datetime, timedelta


class ConfluenceDetector:
    """Tracks multiple signals on same ticker within time window.

    Confluence detection rewards signals that are confirmed by multiple
    independent sources or events within a short time window.

    Confluence Logic:
    - 1 signal -> no bonus (0.0)
    - 2 signals within window -> bonus_2_signals (default 0.10)
    - 3+ signals within window -> bonus_3_signals (default 0.20)
    """

    def __init__(
        self,
        window_minutes: int = 15,
        bonus_2_signals: float = 0.10,
        bonus_3_signals: float = 0.20,
    ):
        """Initialize the confluence detector.

        Args:
            window_minutes: Rolling time window in minutes for counting signals.
            bonus_2_signals: Bonus multiplier for 2 signals within window.
            bonus_3_signals: Bonus multiplier for 3+ signals within window.
        """
        self._window_minutes = window_minutes
        self._bonus_2_signals = bonus_2_signals
        self._bonus_3_signals = bonus_3_signals
        self._signals: dict[str, list[datetime]] = defaultdict(list)

    def record_signal(self, symbol: str, timestamp: datetime) -> None:
        """Record a new signal for a symbol.

        Args:
            symbol: The ticker symbol (e.g., "AAPL").
            timestamp: When the signal occurred.
        """
        self._signals[symbol].append(timestamp)

    def get_confluence_count(self, symbol: str, timestamp: datetime) -> int:
        """Get number of signals for symbol within window.

        Args:
            symbol: The ticker symbol to check.
            timestamp: The reference timestamp for the rolling window.

        Returns:
            Number of signals within [timestamp - window_minutes, timestamp].
        """
        if symbol not in self._signals:
            return 0

        window_start = timestamp - timedelta(minutes=self._window_minutes)
        count = sum(
            1
            for signal_time in self._signals[symbol]
            if window_start <= signal_time <= timestamp
        )
        return count

    def get_bonus(self, symbol: str, timestamp: datetime) -> float:
        """Get confluence bonus (0.0, bonus_2_signals, or bonus_3_signals).

        Args:
            symbol: The ticker symbol to check.
            timestamp: The reference timestamp for the rolling window.

        Returns:
            0.0 for 0-1 signals, bonus_2_signals for 2 signals,
            bonus_3_signals for 3+ signals.
        """
        count = self.get_confluence_count(symbol, timestamp)

        if count >= 3:
            return self._bonus_3_signals
        elif count == 2:
            return self._bonus_2_signals
        else:
            return 0.0

    def cleanup_old_signals(self) -> None:
        """Remove signals outside the window.

        Uses current time as the reference for determining old signals.
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=self._window_minutes)

        for symbol in list(self._signals.keys()):
            self._signals[symbol] = [
                signal_time
                for signal_time in self._signals[symbol]
                if signal_time >= cutoff
            ]
            # Remove empty symbol entries
            if not self._signals[symbol]:
                del self._signals[symbol]
