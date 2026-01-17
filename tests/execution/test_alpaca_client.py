# tests/execution/test_alpaca_client.py
"""Tests for AlpacaClient trading operations."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.execution.alpaca_client import AlpacaClient


class TestAlpacaClientInitialization:
    """Tests for AlpacaClient initialization."""

    @pytest.fixture
    def paper_client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    @pytest.fixture
    def live_client(self):
        return AlpacaClient(
            api_key="live_key",
            secret_key="live_secret",
            paper=False,
        )

    def test_client_initializes_with_paper_mode(self, paper_client):
        """Paper mode should use paper API URL."""
        assert paper_client.paper is True
        assert "paper" in paper_client.base_url
        assert paper_client.base_url == AlpacaClient.PAPER_URL

    def test_client_initializes_with_live_mode(self, live_client):
        """Live mode should use production API URL."""
        assert live_client.paper is False
        assert "paper" not in live_client.base_url
        assert live_client.base_url == AlpacaClient.LIVE_URL

    def test_client_defaults_to_paper_mode(self):
        """Client should default to paper mode for safety."""
        client = AlpacaClient(api_key="key", secret_key="secret")
        assert client.paper is True

    def test_class_constants_are_correct(self):
        """Verify API URL constants."""
        assert AlpacaClient.PAPER_URL == "https://paper-api.alpaca.markets"
        assert AlpacaClient.LIVE_URL == "https://api.alpaca.markets"


class TestAlpacaClientConnection:
    """Tests for connect/disconnect operations."""

    @pytest.fixture
    def client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_connect_initializes_trading_client(self, client):
        """Connect should initialize TradingClient."""
        with patch("src.execution.alpaca_client.TradingClient") as mock_trading:
            with patch("src.execution.alpaca_client.StockHistoricalDataClient") as mock_data:
                await client.connect()

                mock_trading.assert_called_once_with(
                    api_key="test_key",
                    secret_key="test_secret",
                    paper=True,
                )
                mock_data.assert_called_once_with(
                    api_key="test_key",
                    secret_key="test_secret",
                )

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_clients(self, client):
        """Disconnect should clean up client references."""
        with patch("src.execution.alpaca_client.TradingClient"):
            with patch("src.execution.alpaca_client.StockHistoricalDataClient"):
                await client.connect()
                await client.disconnect()

                assert client._trading_client is None
                assert client._data_client is None


class TestAlpacaClientAccount:
    """Tests for account-related operations."""

    @pytest.fixture
    def client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_get_account_returns_account_info(self, client):
        """get_account should return cash, portfolio_value, buying_power."""
        mock_account = MagicMock()
        mock_account.cash = "50000.00"
        mock_account.portfolio_value = "52000.00"
        mock_account.buying_power = "100000.00"

        client._trading_client = MagicMock()
        client._trading_client.get_account.return_value = mock_account

        account = await client.get_account()

        assert account["cash"] == 50000.00
        assert account["portfolio_value"] == 52000.00
        assert account["buying_power"] == 100000.00

    @pytest.mark.asyncio
    async def test_get_account_converts_strings_to_floats(self, client):
        """Account values should be converted from strings to floats."""
        mock_account = MagicMock()
        mock_account.cash = "12345.67"
        mock_account.portfolio_value = "98765.43"
        mock_account.buying_power = "24691.34"

        client._trading_client = MagicMock()
        client._trading_client.get_account.return_value = mock_account

        account = await client.get_account()

        assert isinstance(account["cash"], float)
        assert isinstance(account["portfolio_value"], float)
        assert isinstance(account["buying_power"], float)


class TestAlpacaClientPositions:
    """Tests for position-related operations."""

    @pytest.fixture
    def client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_get_position_returns_position_dict(self, client):
        """get_position should return position details."""
        mock_position = MagicMock()
        mock_position.symbol = "NVDA"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "140.50"
        mock_position.current_price = "142.00"
        mock_position.unrealized_pl = "150.00"
        mock_position.market_value = "14200.00"

        client._trading_client = MagicMock()
        client._trading_client.get_open_position.return_value = mock_position

        position = await client.get_position("NVDA")

        assert position["symbol"] == "NVDA"
        assert position["qty"] == 100
        assert position["avg_entry_price"] == 140.50
        assert position["current_price"] == 142.00
        assert position["unrealized_pl"] == 150.00
        assert position["market_value"] == 14200.00

    @pytest.mark.asyncio
    async def test_get_position_returns_none_when_not_found(self, client):
        """get_position should return None if position doesn't exist."""
        from alpaca.common.exceptions import APIError

        client._trading_client = MagicMock()
        client._trading_client.get_open_position.side_effect = APIError(
            MagicMock(status_code=404)
        )

        position = await client.get_position("NONEXISTENT")

        assert position is None

    @pytest.mark.asyncio
    async def test_get_all_positions_returns_list(self, client):
        """get_all_positions should return list of position dicts."""
        mock_pos1 = MagicMock()
        mock_pos1.symbol = "AAPL"
        mock_pos1.qty = "50"
        mock_pos1.avg_entry_price = "175.00"
        mock_pos1.current_price = "180.00"
        mock_pos1.unrealized_pl = "250.00"
        mock_pos1.market_value = "9000.00"

        mock_pos2 = MagicMock()
        mock_pos2.symbol = "NVDA"
        mock_pos2.qty = "25"
        mock_pos2.avg_entry_price = "140.00"
        mock_pos2.current_price = "145.00"
        mock_pos2.unrealized_pl = "125.00"
        mock_pos2.market_value = "3625.00"

        client._trading_client = MagicMock()
        client._trading_client.get_all_positions.return_value = [mock_pos1, mock_pos2]

        positions = await client.get_all_positions()

        assert len(positions) == 2
        assert positions[0]["symbol"] == "AAPL"
        assert positions[1]["symbol"] == "NVDA"

    @pytest.mark.asyncio
    async def test_get_all_positions_returns_empty_list_when_no_positions(self, client):
        """get_all_positions should return empty list when no positions."""
        client._trading_client = MagicMock()
        client._trading_client.get_all_positions.return_value = []

        positions = await client.get_all_positions()

        assert positions == []


class TestAlpacaClientOrders:
    """Tests for order-related operations."""

    @pytest.fixture
    def client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_submit_market_order(self, client):
        """submit_order should handle market orders."""
        mock_order = MagicMock()
        mock_order.id = "order_123"
        mock_order.status = "accepted"
        mock_order.symbol = "AAPL"
        mock_order.qty = "50"
        mock_order.side = "buy"
        mock_order.type = "market"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None

        client._trading_client = MagicMock()
        client._trading_client.submit_order.return_value = mock_order

        order = await client.submit_order(
            symbol="AAPL",
            qty=50,
            side="buy",
            order_type="market",
        )

        assert order["id"] == "order_123"
        assert order["status"] == "accepted"
        assert order["symbol"] == "AAPL"
        assert order["side"] == "buy"
        assert order["type"] == "market"

    @pytest.mark.asyncio
    async def test_submit_limit_order(self, client):
        """submit_order should handle limit orders with price."""
        mock_order = MagicMock()
        mock_order.id = "order_456"
        mock_order.status = "accepted"
        mock_order.symbol = "NVDA"
        mock_order.qty = "25"
        mock_order.side = "sell"
        mock_order.type = "limit"
        mock_order.limit_price = "150.00"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None

        client._trading_client = MagicMock()
        client._trading_client.submit_order.return_value = mock_order

        order = await client.submit_order(
            symbol="NVDA",
            qty=25,
            side="sell",
            order_type="limit",
            limit_price=150.00,
        )

        assert order["id"] == "order_456"
        assert order["type"] == "limit"
        assert order["limit_price"] == 150.00

    @pytest.mark.asyncio
    async def test_submit_order_uses_correct_request(self, client):
        """submit_order should construct proper OrderRequest."""
        with patch("src.execution.alpaca_client.MarketOrderRequest") as mock_request:
            with patch("src.execution.alpaca_client.OrderSide") as mock_side:
                with patch("src.execution.alpaca_client.TimeInForce") as mock_tif:
                    mock_side.BUY = "buy"
                    mock_tif.DAY = "day"

                    mock_order = MagicMock()
                    mock_order.id = "order_789"
                    mock_order.status = "accepted"
                    mock_order.symbol = "TSLA"
                    mock_order.qty = "10"
                    mock_order.side = "buy"
                    mock_order.type = "market"
                    mock_order.filled_qty = "0"
                    mock_order.filled_avg_price = None

                    client._trading_client = MagicMock()
                    client._trading_client.submit_order.return_value = mock_order

                    await client.submit_order(
                        symbol="TSLA",
                        qty=10,
                        side="buy",
                        order_type="market",
                    )

                    mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """cancel_order should cancel an order by ID."""
        client._trading_client = MagicMock()
        client._trading_client.cancel_order_by_id.return_value = None

        result = await client.cancel_order("order_123")

        client._trading_client.cancel_order_by_id.assert_called_once_with("order_123")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_returns_false_on_error(self, client):
        """cancel_order should return False when order not found."""
        from alpaca.common.exceptions import APIError

        client._trading_client = MagicMock()
        client._trading_client.cancel_order_by_id.side_effect = APIError(
            MagicMock(status_code=404)
        )

        result = await client.cancel_order("nonexistent_order")

        assert result is False


class TestAlpacaClientTimeInForce:
    """Tests for time-in-force handling."""

    @pytest.fixture
    def client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_submit_order_with_gtc_time_in_force(self, client):
        """submit_order should support GTC time in force."""
        mock_order = MagicMock()
        mock_order.id = "order_gtc"
        mock_order.status = "accepted"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.side = "buy"
        mock_order.type = "limit"
        mock_order.time_in_force = "gtc"
        mock_order.limit_price = "150.00"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None

        client._trading_client = MagicMock()
        client._trading_client.submit_order.return_value = mock_order

        order = await client.submit_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            order_type="limit",
            limit_price=150.00,
            time_in_force="gtc",
        )

        assert order["id"] == "order_gtc"
