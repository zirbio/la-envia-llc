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
