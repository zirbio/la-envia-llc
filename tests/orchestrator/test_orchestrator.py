"""Tests for TradingOrchestrator class."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentLabel, SentimentResult
from src.models.social_message import SocialMessage, SourceType
from gate.models import GateCheckResult, GateStatus
from orchestrator.models import OrchestratorState, ProcessResult
from orchestrator.settings import OrchestratorSettings
from orchestrator.trading_orchestrator import TradingOrchestrator
from risk.models import RiskCheckResult
from scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation
from validators.models import TechnicalIndicators, TechnicalValidation, ValidatedSignal, ValidationStatus


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


def make_validated_signal(should_trade: bool = True) -> ValidatedSignal:
    """Create a test validated signal."""
    indicators = TechnicalIndicators(
        rsi=50.0,
        macd_histogram=0.5,
        macd_trend="rising",
        stochastic_k=50.0,
        stochastic_d=50.0,
        adx=25.0,
    )
    validation = TechnicalValidation(
        status=ValidationStatus.PASS if should_trade else ValidationStatus.VETO,
        indicators=indicators,
        veto_reasons=[] if should_trade else ["RSI overbought"],
    )
    return ValidatedSignal(
        message=make_analyzed_message(),
        validation=validation,
    )


def make_recommendation(tier: ScoreTier = ScoreTier.STRONG) -> TradeRecommendation:
    """Create a test trade recommendation."""
    components = ScoreComponents(
        sentiment_score=85.0,
        technical_score=80.0,
        sentiment_weight=0.4,
        technical_weight=0.6,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )
    return TradeRecommendation(
        symbol="AAPL",
        direction=Direction.LONG,
        tier=tier,
        score=85.0,
        position_size_percent=2.0,
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=165.0,
        risk_reward_ratio=3.0,
        components=components,
        reasoning="Strong sentiment",
        timestamp=datetime.now(),
    )


def make_risk_result(approved: bool = True) -> RiskCheckResult:
    """Create a test risk check result."""
    return RiskCheckResult(
        approved=approved,
        adjusted_quantity=10,
        adjusted_value=1500.0,
        rejection_reason=None if approved else "Max daily loss exceeded",
    )


def make_gate_status(is_open: bool = True, factor: float = 1.0) -> GateStatus:
    """Create a test gate status."""
    return GateStatus(
        timestamp=datetime.now(),
        is_open=is_open,
        checks=[GateCheckResult(name="test", passed=is_open, reason=None, data={})],
        position_size_factor=factor,
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


class TestImmediateProcessing:
    """Tests for immediate processing pipeline."""

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

    @pytest.mark.asyncio
    async def test_process_immediate_full_flow(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Full pipeline executes successfully."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.STRONG)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(True)
        mock_components["market_gate"].check.return_value = make_gate_status(True)
        mock_components["trade_executor"].execute.return_value = MagicMock(success=True)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "executed"
        mock_components["trade_executor"].execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_immediate_vetoed(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Vetoed signals don't reach execution."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(False)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "vetoed"
        mock_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_immediate_low_score(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """NO_TRADE tier doesn't reach execution."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.NO_TRADE)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "low_score"
        mock_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_immediate_risk_rejected(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Risk rejection prevents execution."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.STRONG)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(False)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "risk_rejected"
        mock_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_immediate_gate_closed(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Closed gate prevents execution."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.STRONG)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(True)
        mock_components["market_gate"].check.return_value = make_gate_status(False)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "gate_closed"
        mock_components["trade_executor"].execute.assert_not_called()


class TestBatchProcessing:
    """Tests for batch processing logic."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(min_consensus=0.6)

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

    def test_aggregate_sentiment_bullish_consensus(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        """Aggregation calculates bullish consensus correctly."""
        messages = [
            make_analyzed_message(confidence=0.8, label="bullish"),
            make_analyzed_message(confidence=0.7, label="bullish"),
            make_analyzed_message(confidence=0.6, label="bearish"),
        ]

        agg = orchestrator._aggregate_sentiment("AAPL", messages)

        assert agg.consensus_label == "bullish"
        assert agg.bullish_count == 2
        assert agg.bearish_count == 1
        assert agg.consensus_strength == pytest.approx(2/3, rel=0.01)

    def test_aggregate_sentiment_below_consensus_threshold(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        """Aggregation with weak consensus."""
        messages = [
            make_analyzed_message(confidence=0.8, label="bullish"),
            make_analyzed_message(confidence=0.7, label="bearish"),
            make_analyzed_message(confidence=0.6, label="neutral"),
        ]

        agg = orchestrator._aggregate_sentiment("AAPL", messages)

        # Each has 1/3, no strong consensus
        assert agg.consensus_strength == pytest.approx(1/3, rel=0.01)

    def test_group_by_symbol(self, orchestrator: TradingOrchestrator) -> None:
        """Messages are grouped by symbol correctly."""
        msg1 = make_analyzed_message()
        msg1.message.content = "Buy $AAPL now!"
        msg2 = make_analyzed_message()
        msg2.message.content = "Sell $TSLA soon!"
        msg3 = make_analyzed_message()
        msg3.message.content = "Another $AAPL play!"

        orchestrator._message_buffer = [msg1, msg2, msg3]
        grouped = orchestrator._group_by_symbol()

        assert "AAPL" in grouped
        assert "TSLA" in grouped
        assert len(grouped["AAPL"]) == 2
        assert len(grouped["TSLA"]) == 1


class TestGracefulDegradation:
    """Tests for graceful degradation behavior."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(
            continue_without_validator=True,
            gate_fail_safe_closed=True,
        )

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

    @pytest.mark.asyncio
    async def test_validator_error_continues_when_enabled(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Validator error doesn't stop processing when continue_without_validator=True."""
        mock_components["technical_validator"].validate.side_effect = Exception("Validator down")
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.STRONG)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(True)
        mock_components["market_gate"].check.return_value = make_gate_status(True)
        mock_components["trade_executor"].execute.return_value = MagicMock(success=True)

        result = await orchestrator._process_immediate(make_analyzed_message())

        # Should continue and execute despite validator error
        assert result.status == "executed"

    @pytest.mark.asyncio
    async def test_gate_error_closes_when_fail_safe(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Gate error treated as closed when gate_fail_safe_closed=True."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.STRONG)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(True)
        mock_components["market_gate"].check.side_effect = Exception("Gate API down")

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "gate_closed"
        mock_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_validator_error_stops_when_disabled(
        self, mock_components: dict
    ) -> None:
        """Validator error stops processing when continue_without_validator=False."""
        settings = OrchestratorSettings(continue_without_validator=False)
        orchestrator = TradingOrchestrator(
            collector_manager=mock_components["collector_manager"],
            analyzer_manager=mock_components["analyzer_manager"],
            technical_validator=mock_components["technical_validator"],
            signal_scorer=mock_components["signal_scorer"],
            risk_manager=mock_components["risk_manager"],
            market_gate=mock_components["market_gate"],
            trade_executor=mock_components["trade_executor"],
            settings=settings,
        )

        mock_components["technical_validator"].validate.side_effect = Exception("Validator down")

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "error"
        assert "Validation error" in result.error

    @pytest.mark.asyncio
    async def test_gate_error_returns_error_when_fail_safe_disabled(
        self, mock_components: dict
    ) -> None:
        """Gate error returns error when gate_fail_safe_closed=False."""
        settings = OrchestratorSettings(gate_fail_safe_closed=False)
        orchestrator = TradingOrchestrator(
            collector_manager=mock_components["collector_manager"],
            analyzer_manager=mock_components["analyzer_manager"],
            technical_validator=mock_components["technical_validator"],
            signal_scorer=mock_components["signal_scorer"],
            risk_manager=mock_components["risk_manager"],
            market_gate=mock_components["market_gate"],
            trade_executor=mock_components["trade_executor"],
            settings=settings,
        )

        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(ScoreTier.STRONG)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(True)
        mock_components["market_gate"].check.side_effect = Exception("Gate API down")

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "error"
        assert "Gate error" in result.error
        mock_components["trade_executor"].execute.assert_not_called()
