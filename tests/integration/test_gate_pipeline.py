# tests/integration/test_gate_pipeline.py
"""Integration tests for market gate with execution pipeline."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.execution.trade_executor import TradeExecutor
from src.gate.market_gate import MarketGate, MarketGateSettings
from src.gate.models import GateStatus
from src.risk.models import RiskCheckResult
from src.risk.risk_manager import RiskManager
from src.scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation


ET = ZoneInfo("America/New_York")


def make_recommendation(
    symbol: str = "AAPL",
    direction: Direction = Direction.LONG,
    entry_price: float = 150.00,
    stop_loss: float = 145.00,
    take_profit: float = 165.00,
) -> TradeRecommendation:
    """Helper to create TradeRecommendation for tests."""
    return TradeRecommendation(
        symbol=symbol,
        direction=direction,
        score=85.0,
        tier=ScoreTier.STRONG,
        position_size_percent=100.0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=2.0,
        components=ScoreComponents(
            sentiment_score=90.0,
            technical_score=80.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.0,
            credibility_multiplier=1.0,
            time_factor=1.0,
        ),
        reasoning="Strong sentiment analysis",
        timestamp=datetime.now(),
    )


class TestGatePipelineIntegration:
    """Integration tests for gate -> risk -> execution flow."""

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
        return make_recommendation()

    @pytest.mark.asyncio
    async def test_full_flow_gate_open(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Full flow executes when gate is open."""
        # Setup mocks
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5
        mock_alpaca.submit_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 10,
        }
        mock_alpaca.get_all_positions.return_value = []

        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # Check gate during market hours
        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is True
        assert gate_status.position_size_factor == 1.0

        # Check risk
        risk_result = risk_manager.check_trade(recommendation, 10, 150.0)
        assert risk_result.approved is True

        # Execute
        executor = TradeExecutor(mock_alpaca, risk_manager)
        exec_result = await executor.execute(recommendation, risk_result, gate_status)

        assert exec_result.success is True
        assert exec_result.order_id == "order-123"
        assert exec_result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_full_flow_gate_closed_blocks_execution(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Execution blocked when gate is closed."""
        mock_alpaca = AsyncMock()
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 35.0  # VIX blocked (above 30)

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

        # Verify no order was submitted
        mock_alpaca.submit_order.assert_not_called()

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
        mock_alpaca.submit_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 5,
        }
        mock_alpaca.get_all_positions.return_value = []

        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 27.0  # Elevated (25-30 range)

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is True
        assert gate_status.position_size_factor == 0.5

        risk_result = risk_manager.check_trade(recommendation, 10, 150.0)

        executor = TradeExecutor(mock_alpaca, risk_manager)
        await executor.execute(recommendation, risk_result, gate_status)

        # Verify reduced quantity was used (10 * 0.5 = 5)
        call_args = mock_alpaca.submit_order.call_args
        assert call_args.kwargs["qty"] == 5

    @pytest.mark.asyncio
    async def test_gate_closed_outside_trading_hours(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Gate is closed outside trading hours."""
        mock_alpaca = AsyncMock()
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # Check before market open (9:00 AM ET)
        test_time = datetime(2026, 1, 17, 9, 0, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is False
        failed_checks = gate_status.get_failed_checks()
        assert any(check.name == "trading_hours" for check in failed_checks)

    @pytest.mark.asyncio
    async def test_gate_closed_during_lunch(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Gate is closed during lunch hours when avoid_lunch is enabled."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 600_000},
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        settings = MarketGateSettings(avoid_lunch=True)
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        # Check during lunch hours (12:00 PM ET)
        test_time = datetime(2026, 1, 17, 12, 0, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is False
        failed_checks = gate_status.get_failed_checks()
        assert any(check.name == "trading_hours" for check in failed_checks)

    @pytest.mark.asyncio
    async def test_gate_closed_low_volume(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Gate is closed when volume is insufficient."""
        mock_alpaca = AsyncMock()
        mock_alpaca.get_latest_bar.side_effect = [
            {"symbol": "SPY", "volume": 100_000},  # Below minimum 500k
            {"symbol": "QQQ", "volume": 400_000},
        ]
        mock_alpaca.get_bars.return_value = [{"high": 455, "low": 450, "close": 453}]
        mock_alpaca.get_atr.return_value = 2.5
        mock_vix_fetcher = MagicMock()
        mock_vix_fetcher.fetch_vix.return_value = 18.0

        settings = MarketGateSettings()
        gate = MarketGate(mock_alpaca, settings, mock_vix_fetcher)

        test_time = datetime(2026, 1, 17, 10, 30, tzinfo=ET)
        gate_status = await gate.check(current_time=test_time)

        assert gate_status.is_open is False
        failed_checks = gate_status.get_failed_checks()
        assert any(check.name == "volume" for check in failed_checks)

    @pytest.mark.asyncio
    async def test_gate_no_status_executes_normally(
        self, risk_manager: RiskManager, recommendation: TradeRecommendation
    ) -> None:
        """Execution proceeds when gate_status is None (backwards compatibility)."""
        mock_alpaca = AsyncMock()
        mock_alpaca.submit_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 10,
        }
        mock_alpaca.get_all_positions.return_value = []

        risk_result = risk_manager.check_trade(recommendation, 10, 150.0)
        assert risk_result.approved is True

        executor = TradeExecutor(mock_alpaca, risk_manager)
        exec_result = await executor.execute(recommendation, risk_result, gate_status=None)

        assert exec_result.success is True
        assert exec_result.quantity == 10
