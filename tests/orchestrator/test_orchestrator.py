"""Tests for TradingOrchestrator class."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentLabel, SentimentResult
from src.models.social_message import SocialMessage, SourceType
from orchestrator.models import OrchestratorState, ProcessResult
from orchestrator.settings import OrchestratorSettings
from orchestrator.trading_orchestrator import TradingOrchestrator


def make_social_message(content: str = "Test message") -> SocialMessage:
    """Create a test social message."""
    return SocialMessage(
        source=SourceType.TWITTER,
        source_id="test_123",
        content=content,
        author="test_user",
        timestamp=datetime.now(),
    )


def make_analyzed_message(
    confidence: float = 0.9,
    label: str = "bullish",
) -> AnalyzedMessage:
    """Create a test analyzed message."""
    # Convert string label to SentimentLabel enum
    label_enum = SentimentLabel(label)
    sentiment = SentimentResult(
        label=label_enum,
        score=confidence if label == "bullish" else 0.1,
        confidence=confidence,
    )
    return AnalyzedMessage(
        message=make_social_message(),
        sentiment=sentiment,
    )


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


class TestHighSignalDetection:
    """Tests for high-signal detection."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(immediate_threshold=0.85)

    @pytest.fixture
    def orchestrator(self, settings: OrchestratorSettings) -> TradingOrchestrator:
        return TradingOrchestrator(
            collector_manager=MagicMock(),
            analyzer_manager=AsyncMock(),
            technical_validator=AsyncMock(),
            signal_scorer=MagicMock(),
            risk_manager=MagicMock(),
            market_gate=AsyncMock(),
            trade_executor=AsyncMock(),
            settings=settings,
        )

    def test_is_high_signal_true_for_high_confidence(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        msg = make_analyzed_message(confidence=0.9, label="bullish")
        assert orchestrator._is_high_signal(msg) is True

    def test_is_high_signal_false_for_low_confidence(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        msg = make_analyzed_message(confidence=0.7, label="bullish")
        assert orchestrator._is_high_signal(msg) is False

    def test_is_high_signal_false_for_neutral(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        msg = make_analyzed_message(confidence=0.95, label="neutral")
        assert orchestrator._is_high_signal(msg) is False

    def test_is_high_signal_at_threshold(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        msg = make_analyzed_message(confidence=0.85, label="bearish")
        assert orchestrator._is_high_signal(msg) is True


class TestMessageRouting:
    """Tests for message routing logic."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(immediate_threshold=0.85)

    @pytest.fixture
    def mock_analyzer(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def orchestrator(self, settings: OrchestratorSettings, mock_analyzer: AsyncMock) -> TradingOrchestrator:
        return TradingOrchestrator(
            collector_manager=MagicMock(),
            analyzer_manager=mock_analyzer,
            technical_validator=AsyncMock(),
            signal_scorer=MagicMock(),
            risk_manager=MagicMock(),
            market_gate=AsyncMock(),
            trade_executor=AsyncMock(),
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_on_message_routes_high_signal_to_immediate(
        self, orchestrator: TradingOrchestrator, mock_analyzer: AsyncMock
    ) -> None:
        """High-signal messages are processed immediately."""
        high_signal_msg = make_analyzed_message(confidence=0.9, label="bullish")
        mock_analyzer.analyze.return_value = high_signal_msg

        # Mock the immediate processing
        orchestrator._process_immediate = AsyncMock(return_value=ProcessResult(status="executed"))

        result = await orchestrator._on_message(make_social_message())

        orchestrator._process_immediate.assert_called_once()
        assert result.status == "executed"

    @pytest.mark.asyncio
    async def test_on_message_routes_low_signal_to_buffer(
        self, orchestrator: TradingOrchestrator, mock_analyzer: AsyncMock
    ) -> None:
        """Low-signal messages are added to buffer."""
        low_signal_msg = make_analyzed_message(confidence=0.5, label="bullish")
        mock_analyzer.analyze.return_value = low_signal_msg

        result = await orchestrator._on_message(make_social_message())

        assert len(orchestrator._message_buffer) == 1
        assert result.status == "buffered"

    @pytest.mark.asyncio
    async def test_on_message_handles_analyzer_error(
        self, orchestrator: TradingOrchestrator, mock_analyzer: AsyncMock
    ) -> None:
        """Analyzer errors are handled gracefully."""
        mock_analyzer.analyze.side_effect = Exception("Analysis failed")

        result = await orchestrator._on_message(make_social_message())

        assert result.status == "error"
        assert "Analysis failed" in result.error
