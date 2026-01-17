# src/execution/alpaca_client.py
"""Alpaca trading client for order execution and account management."""

from typing import Optional

import pandas as pd

# Import at module level for mocking in tests
try:
    from alpaca.common.exceptions import APIError
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
except ImportError:
    TradingClient = None
    StockHistoricalDataClient = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    OrderSide = None
    TimeInForce = None
    APIError = Exception
    StockBarsRequest = None
    TimeFrame = None
    TimeFrameUnit = None


class AlpacaClient:
    """Unified client for Alpaca Trading API.

    Provides methods for account management, position tracking,
    and order execution using the Alpaca brokerage API.

    Attributes:
        PAPER_URL: URL for paper trading API.
        LIVE_URL: URL for live trading API.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        """Initialize AlpacaClient.

        Args:
            api_key: Alpaca API key.
            secret_key: Alpaca secret key.
            paper: If True, use paper trading (default). If False, use live trading.
        """
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._base_url = self.PAPER_URL if paper else self.LIVE_URL
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None

    @property
    def paper(self) -> bool:
        """Return whether client is in paper trading mode."""
        return self._paper

    @property
    def base_url(self) -> str:
        """Return the base API URL being used."""
        return self._base_url

    async def connect(self) -> None:
        """Initialize Alpaca clients.

        Creates TradingClient and StockHistoricalDataClient instances.

        Raises:
            RuntimeError: If alpaca-py is not installed.
        """
        if TradingClient is None:
            raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")

        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

    async def disconnect(self) -> None:
        """Clean up Alpaca client resources."""
        self._trading_client = None
        self._data_client = None

    async def get_account(self) -> dict:
        """Get account information.

        Returns:
            Dictionary containing:
                - cash: Available cash balance (float)
                - portfolio_value: Total portfolio value (float)
                - buying_power: Available buying power (float)
        """
        account = self._trading_client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
        }

    async def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "NVDA").

        Returns:
            Dictionary containing position details, or None if no position exists.
            Keys: symbol, qty, avg_entry_price, current_price,
                  unrealized_pl, market_value
        """
        try:
            position = self._trading_client.get_open_position(symbol)
            return self._position_to_dict(position)
        except APIError:
            return None

    async def get_all_positions(self) -> list[dict]:
        """Get all open positions.

        Returns:
            List of position dictionaries. Empty list if no positions.
        """
        positions = self._trading_client.get_all_positions()
        return [self._position_to_dict(pos) for pos in positions]

    def _position_to_dict(self, position) -> dict:
        """Convert Alpaca position object to dictionary.

        Args:
            position: Alpaca Position object.

        Returns:
            Dictionary with position details.
        """
        return {
            "symbol": position.symbol,
            "qty": int(position.qty),
            "avg_entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "unrealized_pl": float(position.unrealized_pl),
            "market_value": float(position.market_value),
        }

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> dict:
        """Submit a trading order.

        Args:
            symbol: Stock symbol (e.g., "AAPL").
            qty: Number of shares.
            side: Order side ("buy" or "sell").
            order_type: Order type ("market" or "limit").
            limit_price: Limit price for limit orders.
            time_in_force: Time in force ("day", "gtc", etc.). Defaults to "day".

        Returns:
            Dictionary containing order details:
                - id: Order ID
                - status: Order status
                - symbol: Stock symbol
                - qty: Quantity
                - side: Order side
                - type: Order type
                - limit_price: Limit price (if applicable)
                - filled_qty: Filled quantity
                - filled_avg_price: Average fill price
        """
        # Determine order side
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Determine time in force
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

        # Create order request based on type
        if order_type.lower() == "market":
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
        else:  # limit order
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )

        order = self._trading_client.submit_order(order_request)
        return self._order_to_dict(order)

    def _order_to_dict(self, order) -> dict:
        """Convert Alpaca order object to dictionary.

        Args:
            order: Alpaca Order object.

        Returns:
            Dictionary with order details.
        """
        result = {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": int(order.qty),
            "side": str(order.side),
            "type": str(order.type),
            "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": (
                float(order.filled_avg_price) if order.filled_avg_price else None
            ),
        }

        # Add limit price if present
        if hasattr(order, "limit_price") and order.limit_price is not None:
            result["limit_price"] = float(order.limit_price)

        return result

    async def get_order(self, order_id: str) -> dict:
        """Get order details by ID.

        Args:
            order_id: The order ID to retrieve.

        Returns:
            Dictionary containing order details.

        Raises:
            APIError: If order cannot be retrieved.
        """
        order = self._trading_client.get_order_by_id(order_id)
        return self._order_to_dict(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID.

        Args:
            order_id: The order ID to cancel.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        try:
            self._trading_client.cancel_order_by_id(order_id)
            return True
        except APIError:
            return False

    def get_bars(
        self,
        symbol: str,
        limit: int = 50,
        timeframe: str = "5Min",
    ) -> pd.DataFrame:
        """Get historical bars for a symbol.

        Args:
            symbol: Stock symbol (e.g., "NVDA")
            limit: Number of bars to fetch
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")

        Returns:
            DataFrame with OHLCV data (columns: open, high, low, close, volume)
        """
        # Map timeframe string to TimeFrame object
        timeframe_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }

        # Get TimeFrame object or default to 5Min
        tf = timeframe_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

        # Create StockBarsRequest
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            limit=limit,
        )

        # Get bars from data client
        bars_df = self._data_client.get_stock_bars(request)

        # Handle empty DataFrame
        if bars_df.empty:
            return pd.DataFrame()

        # Handle MultiIndex DataFrame - drop symbol level if present
        if isinstance(bars_df.index, pd.MultiIndex):
            bars_df = bars_df.droplevel("symbol")

        return bars_df
