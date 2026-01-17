"""Tests for TradingOrchestrator class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.models import OrchestratorState
from orchestrator.settings import OrchestratorSettings
from orchestrator.trading_orchestrator import TradingOrchestrator


class TestTradingOrchestratorCore:
    """Tests for core orchestrator functionality."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings()

    @pytest.fixture
    def mock_components(self) -> dict:
        return {
            "collector_manager": MagicMock(),
            "analyzer_manager": AsyncMock(),
            "technical_validator": AsyncMock(),
            "signal_scorer": MagicMock(),
            "risk_manager": MagicMock(),
            "market_gate": AsyncMock(),
            "trade_executor": AsyncMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_components: dict, settings: OrchestratorSettings) -> TradingOrchestrator:
        return TradingOrchestrator(
            collector_manager=mock_components["collector_manager"],
            analyzer_manager=mock_components["analyzer_manager"],
            technical_validator=mock_components["technical_validator"],
            signal_scorer=mock_components["signal_scorer"],
            risk_manager=mock_components["risk_manager"],
            market_gate=mock_components["market_gate"],
            trade_executor=mock_components["trade_executor"],
            settings=settings,
        )

    def test_initial_state_is_stopped(self, orchestrator: TradingOrchestrator) -> None:
        assert orchestrator.state == OrchestratorState.STOPPED

    def test_is_running_false_when_stopped(self, orchestrator: TradingOrchestrator) -> None:
        assert orchestrator.is_running is False

    def test_buffer_initially_empty(self, orchestrator: TradingOrchestrator) -> None:
        assert len(orchestrator._message_buffer) == 0


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items: list):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class TestTradingOrchestratorStartStop:
    """Tests for start/stop functionality."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings()

    @pytest.fixture
    def mock_components(self) -> dict:
        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.stream_all = MagicMock(return_value=AsyncIteratorMock([]))
        collector.add_callback = MagicMock()

        return {
            "collector_manager": collector,
            "analyzer_manager": AsyncMock(),
            "technical_validator": AsyncMock(),
            "signal_scorer": MagicMock(),
            "risk_manager": MagicMock(),
            "market_gate": AsyncMock(),
            "trade_executor": AsyncMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_components: dict, settings: OrchestratorSettings) -> TradingOrchestrator:
        return TradingOrchestrator(
            collector_manager=mock_components["collector_manager"],
            analyzer_manager=mock_components["analyzer_manager"],
            technical_validator=mock_components["technical_validator"],
            signal_scorer=mock_components["signal_scorer"],
            risk_manager=mock_components["risk_manager"],
            market_gate=mock_components["market_gate"],
            trade_executor=mock_components["trade_executor"],
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_start_changes_state_to_running(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        await orchestrator.start()
        assert orchestrator.state == OrchestratorState.RUNNING
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_start_connects_collectors(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        await orchestrator.start()
        mock_components["collector_manager"].connect_all.assert_called_once()
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stop_changes_state_to_stopped(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        await orchestrator.start()
        await orchestrator.stop()
        assert orchestrator.state == OrchestratorState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_disconnects_collectors(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        await orchestrator.start()
        await orchestrator.stop()
        mock_components["collector_manager"].disconnect_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_raises_if_already_running(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        await orchestrator.start()
        with pytest.raises(RuntimeError, match="already running"):
            await orchestrator.start()
        await orchestrator.stop()
