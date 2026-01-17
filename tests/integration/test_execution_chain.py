"""Integration tests for the execution chain.

Tests the execution path: Signals → Gate → Risk → Execution → Journal
"""

from datetime import datetime, time, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from gate.market_gate import MarketGate, MarketGateSettings
from gate.models import GateCheckResult, GateStatus
from gate.vix_fetcher import VixFetcher
from src.execution.alpaca_client import AlpacaClient
from src.execution.models import ExecutionResult, TrackedPosition
from src.execution.trade_executor import TradeExecutor
from src.journal.journal_manager import JournalManager
from src.journal.settings import JournalSettings
from src.risk.risk_manager import RiskManager
from src.scoring.models import (
    Direction,
    ScoreComponents,
    ScoreTier,
    TradeRecommendation,
)


ET = ZoneInfo("America/New_York")


class TestExecutionChainIntegration:
    """Integration tests for the full execution chain."""

    @pytest.fixture
    def mock_alpaca_client(self) -> AsyncMock:
        """Create mocked Alpaca client."""
        client = AsyncMock()

        # Mock get_latest_bar for volume checks
        client.get_latest_bar.return_value = {"volume": 1_000_000}

        # Mock get_bars for choppy market checks
        client.get_bars.return_value = [
            {"high": 500.0, "low": 495.0}
        ]

        # Mock get_atr
        client.get_atr.return_value = 2.0

        # Mock submit_order
        client.submit_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 100.0,
            "filled_qty": 5,
        }

        # Mock get_all_positions
        client.get_all_positions.return_value = []

        # Mock get_order
        client.get_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 106.0,
        }

        return client

    @pytest.fixture
    def mock_vix_fetcher(self) -> MagicMock:
        """Create mocked VIX fetcher."""
        fetcher = MagicMock(spec=VixFetcher)
        fetcher.fetch_vix.return_value = 15.0  # Normal VIX
        return fetcher

    @pytest.fixture
    def gate_settings(self) -> MarketGateSettings:
        """Create gate settings."""
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
    def market_gate(
        self,
        mock_alpaca_client: AsyncMock,
        gate_settings: MarketGateSettings,
        mock_vix_fetcher: MagicMock,
    ) -> MarketGate:
        """Create market gate with mocked dependencies."""
        return MarketGate(
            alpaca_client=mock_alpaca_client,
            settings=gate_settings,
            vix_fetcher=mock_vix_fetcher,
        )

    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        """Create risk manager."""
        return RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
            unrealized_warning_threshold=300.0,
        )

    @pytest.fixture
    def trade_executor(
        self,
        mock_alpaca_client: AsyncMock,
        risk_manager: RiskManager,
    ) -> TradeExecutor:
        """Create trade executor with mocked Alpaca client."""
        return TradeExecutor(
            alpaca_client=mock_alpaca_client,
            risk_manager=risk_manager,
        )

    @pytest.fixture
    def journal_settings(self) -> JournalSettings:
        """Create journal settings."""
        return JournalSettings(
            enabled=True,
            base_path="/tmp/test_journal",
            retention_days=30,
        )

    @pytest.fixture
    def journal_manager(self, journal_settings: JournalSettings) -> JournalManager:
        """Create journal manager."""
        return JournalManager(settings=journal_settings)

    @pytest.fixture
    def strong_recommendation(self) -> TradeRecommendation:
        """Create a strong LONG recommendation."""
        return TradeRecommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=100.0,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            risk_reward_ratio=3.0,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.1,
                credibility_multiplier=1.2,
                time_factor=1.0,
            ),
            reasoning="Strong bullish signal with high volume",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_signal_to_gate_open_flow(
        self,
        market_gate: MarketGate,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test signal flowing through an open gate.

        Flow: Signal → Gate (OPEN) → Approved
        """
        # Check gate during trading hours
        test_time = datetime(2024, 1, 15, 14, 30, tzinfo=ET)  # 2:30 PM ET

        gate_status = await market_gate.check(current_time=test_time)

        # Verify gate is open
        assert gate_status.is_open
        assert gate_status.position_size_factor == 1.0
        assert all(check.passed for check in gate_status.checks)

    @pytest.mark.asyncio
    async def test_signal_to_gate_closed_flow(
        self,
        market_gate: MarketGate,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test signal flowing through a closed gate.

        Flow: Signal → Gate (CLOSED) → Blocked
        """
        # Check gate during lunch hours
        test_time = datetime(2024, 1, 15, 12, 0, tzinfo=ET)  # 12:00 PM ET (lunch)

        gate_status = await market_gate.check(current_time=test_time)

        # Verify gate is closed
        assert not gate_status.is_open
        assert gate_status.position_size_factor == 0.0

        # Verify lunch hours check failed
        failed_checks = gate_status.get_failed_checks()
        assert len(failed_checks) > 0
        assert any("lunch" in check.reason.lower() for check in failed_checks)

    @pytest.mark.asyncio
    async def test_gate_to_risk_flow(
        self,
        market_gate: MarketGate,
        risk_manager: RiskManager,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test gate passing to risk validation.

        Flow: Gate (OPEN) → Risk Check → Approved
        """
        # Check gate during trading hours
        test_time = datetime(2024, 1, 15, 14, 30, tzinfo=ET)
        gate_status = await market_gate.check(current_time=test_time)

        assert gate_status.is_open

        # Pass to risk manager
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )

        # Verify risk check approved
        assert risk_result.approved
        assert risk_result.adjusted_quantity == 5
        assert risk_result.adjusted_value == 500.0
        assert risk_result.rejection_reason is None

    @pytest.mark.asyncio
    async def test_risk_to_execution_flow(
        self,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test risk approval flowing to execution.

        Flow: Risk (APPROVED) → Execution → Order Submitted
        """
        # Check risk
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )

        assert risk_result.approved

        # Execute trade
        execution_result = await trade_executor.execute(
            recommendation=strong_recommendation,
            risk_result=risk_result,
        )

        # Verify execution success
        assert execution_result.success
        assert execution_result.order_id == "order-123"
        assert execution_result.symbol == "NVDA"
        assert execution_result.side == "buy"
        assert execution_result.quantity == 5
        assert execution_result.filled_price == 100.0
        assert execution_result.error_message is None

    @pytest.mark.asyncio
    async def test_risk_rejection_blocks_execution(
        self,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test risk rejection preventing execution.

        Flow: Risk (REJECTED) → Execution → Blocked
        """
        # Request excessive position size
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=15,  # 15 * $100 = $1500 > $1000 limit
            current_price=100.0,
        )

        assert not risk_result.approved

        # Try to execute
        execution_result = await trade_executor.execute(
            recommendation=strong_recommendation,
            risk_result=risk_result,
        )

        # Verify execution blocked
        assert not execution_result.success
        assert execution_result.order_id is None
        assert execution_result.quantity == 0
        assert "position value" in execution_result.error_message.lower()

    @pytest.mark.asyncio
    async def test_gate_closed_blocks_execution(
        self,
        market_gate: MarketGate,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test closed gate preventing execution.

        Flow: Gate (CLOSED) → Risk (APPROVED) → Execution → Blocked by Gate
        """
        # Check gate during lunch
        test_time = datetime(2024, 1, 15, 12, 0, tzinfo=ET)
        gate_status = await market_gate.check(current_time=test_time)

        assert not gate_status.is_open

        # Risk check passes
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )

        assert risk_result.approved

        # Execute with closed gate
        execution_result = await trade_executor.execute(
            recommendation=strong_recommendation,
            risk_result=risk_result,
            gate_status=gate_status,
        )

        # Verify execution blocked by gate
        assert not execution_result.success
        assert execution_result.order_id is None
        assert "gate closed" in execution_result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execution_to_journal_flow(
        self,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        journal_manager: JournalManager,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test execution flowing to journal.

        Flow: Execution (SUCCESS) → Journal → Trade Logged
        """
        # Approve and execute trade
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )

        execution_result = await trade_executor.execute(
            recommendation=strong_recommendation,
            risk_result=risk_result,
        )

        assert execution_result.success

        # Get tracked position
        positions = trade_executor.get_tracked_positions()
        assert "NVDA" in positions
        tracked_position = positions["NVDA"]

        # Log to journal
        trade_id = await journal_manager.on_trade_opened(
            position=tracked_position,
            recommendation=strong_recommendation,
        )

        # Verify journal entry created
        assert trade_id is not None
        assert isinstance(trade_id, str)

    @pytest.mark.asyncio
    async def test_full_chain_success_path(
        self,
        market_gate: MarketGate,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        journal_manager: JournalManager,
        strong_recommendation: TradeRecommendation,
    ) -> None:
        """Test complete chain with successful execution.

        Flow: Signal → Gate → Risk → Execution → Journal (All Pass)
        """
        # Step 1: Check gate
        test_time = datetime(2024, 1, 15, 14, 30, tzinfo=ET)
        gate_status = await market_gate.check(current_time=test_time)
        assert gate_status.is_open

        # Step 2: Check risk
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert risk_result.approved

        # Step 3: Execute trade
        execution_result = await trade_executor.execute(
            recommendation=strong_recommendation,
            risk_result=risk_result,
            gate_status=gate_status,
        )
        assert execution_result.success

        # Step 4: Log to journal
        positions = trade_executor.get_tracked_positions()
        tracked_position = positions["NVDA"]

        trade_id = await journal_manager.on_trade_opened(
            position=tracked_position,
            recommendation=strong_recommendation,
        )
        assert trade_id is not None

        # Verify complete chain success
        assert execution_result.order_id == "order-123"
        assert tracked_position.symbol == "NVDA"
        assert tracked_position.quantity == 5
        assert tracked_position.entry_price == 100.0

    @pytest.mark.asyncio
    async def test_full_chain_with_vix_elevation(
        self,
        market_gate: MarketGate,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        strong_recommendation: TradeRecommendation,
        mock_vix_fetcher: MagicMock,
    ) -> None:
        """Test chain with elevated VIX reducing position size.

        Flow: Signal → Gate (VIX ELEVATED) → Risk → Execution (Reduced Size)
        """
        # Set elevated VIX (25.0-30.0 range)
        mock_vix_fetcher.fetch_vix.return_value = 27.0

        # Step 1: Check gate
        test_time = datetime(2024, 1, 15, 14, 30, tzinfo=ET)
        gate_status = await market_gate.check(current_time=test_time)

        # Verify gate is open but with reduced position size factor
        assert gate_status.is_open
        assert gate_status.position_size_factor == 0.5  # Elevated size factor

        # Step 2: Check risk
        risk_result = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=10,  # Request 10 shares
            current_price=100.0,
        )
        assert risk_result.approved
        assert risk_result.adjusted_quantity == 10

        # Step 3: Execute trade (should reduce quantity)
        execution_result = await trade_executor.execute(
            recommendation=strong_recommendation,
            risk_result=risk_result,
            gate_status=gate_status,
        )

        # Verify execution succeeded with reduced quantity
        assert execution_result.success
        # 10 shares * 0.5 factor = 5 shares
        assert execution_result.quantity == 5

    @pytest.mark.asyncio
    async def test_neutral_direction_rejected(
        self,
        market_gate: MarketGate,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
    ) -> None:
        """Test NEUTRAL direction rejected by risk manager.

        Flow: Signal (NEUTRAL) → Gate → Risk → REJECTED
        """
        # Create NEUTRAL recommendation
        neutral_recommendation = TradeRecommendation(
            symbol="SPY",
            direction=Direction.NEUTRAL,
            score=50.0,
            tier=ScoreTier.WEAK,
            position_size_percent=0.0,
            entry_price=450.0,
            stop_loss=448.0,
            take_profit=452.0,
            risk_reward_ratio=1.0,
            components=ScoreComponents(
                sentiment_score=50.0,
                technical_score=50.0,
                sentiment_weight=0.5,
                technical_weight=0.5,
                confluence_bonus=0.0,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="No clear direction",
            timestamp=datetime.now(timezone.utc),
        )

        # Gate passes
        test_time = datetime(2024, 1, 15, 14, 30, tzinfo=ET)
        gate_status = await market_gate.check(current_time=test_time)
        assert gate_status.is_open

        # Risk check rejects NEUTRAL
        risk_result = risk_manager.check_trade(
            recommendation=neutral_recommendation,
            requested_quantity=5,
            current_price=450.0,
        )

        assert not risk_result.approved
        assert "neutral" in risk_result.rejection_reason.lower()

        # Execution blocked
        execution_result = await trade_executor.execute(
            recommendation=neutral_recommendation,
            risk_result=risk_result,
            gate_status=gate_status,
        )

        assert not execution_result.success

    @pytest.mark.asyncio
    async def test_short_direction_execution_flow(
        self,
        market_gate: MarketGate,
        risk_manager: RiskManager,
        trade_executor: TradeExecutor,
        mock_alpaca_client: AsyncMock,
    ) -> None:
        """Test SHORT direction flows correctly through chain.

        Flow: Signal (SHORT) → Gate → Risk → Execution (SELL order)
        """
        # Create SHORT recommendation
        short_recommendation = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            score=75.0,
            tier=ScoreTier.STRONG,
            position_size_percent=80.0,
            entry_price=200.0,
            stop_loss=205.0,
            take_profit=190.0,
            risk_reward_ratio=2.0,
            components=ScoreComponents(
                sentiment_score=70.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.05,
                credibility_multiplier=1.1,
                time_factor=1.0,
            ),
            reasoning="Strong bearish signal",
            timestamp=datetime.now(timezone.utc),
        )

        # Update mock for SHORT order
        mock_alpaca_client.submit_order.return_value = {
            "id": "short-order-456",
            "filled_avg_price": 200.0,
            "filled_qty": 3,
        }

        # Gate passes
        test_time = datetime(2024, 1, 15, 14, 30, tzinfo=ET)
        gate_status = await market_gate.check(current_time=test_time)
        assert gate_status.is_open

        # Risk approves
        risk_result = risk_manager.check_trade(
            recommendation=short_recommendation,
            requested_quantity=3,
            current_price=200.0,
        )
        assert risk_result.approved

        # Execute SHORT order
        execution_result = await trade_executor.execute(
            recommendation=short_recommendation,
            risk_result=risk_result,
            gate_status=gate_status,
        )

        # Verify SHORT execution
        assert execution_result.success
        assert execution_result.side == "sell"
        assert execution_result.order_id == "short-order-456"

        # Verify Alpaca was called with 'sell'
        mock_alpaca_client.submit_order.assert_called_once()
        call_kwargs = mock_alpaca_client.submit_order.call_args[1]
        assert call_kwargs["side"] == "sell"
        assert call_kwargs["symbol"] == "TSLA"
