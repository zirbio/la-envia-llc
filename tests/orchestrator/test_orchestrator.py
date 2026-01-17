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
