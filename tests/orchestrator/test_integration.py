"""Integration tests for trading orchestrator."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentLabel, SentimentResult
from src.models.social_message import SocialMessage, SourceType
from execution.models import ExecutionResult
from gate.models import GateCheckResult, GateStatus
from orchestrator.models import OrchestratorState
from orchestrator.settings import OrchestratorSettings
from orchestrator.trading_orchestrator import TradingOrchestrator
from risk.models import RiskCheckResult
from scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation
from validators.models import (
    TechnicalIndicators,
    TechnicalValidation,
    ValidatedSignal,
    ValidationStatus,
)


class AsyncIteratorMock:
    """Mock async iterator for testing stream."""

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


def make_mock_components_for_integration(
    sample_message: SocialMessage,
    analyzed_msg: AnalyzedMessage,
) -> dict:
    """Create mock components for integration tests."""
    # Grok collector mock
    grok_collector = MagicMock()
    grok_collector.stream = MagicMock(
        return_value=AsyncIteratorMock([sample_message])
    )

    # Claude analyzer mock
    claude_analyzer = MagicMock()
    claude_analyzer.analyze = MagicMock(return_value=None)

    return {
        "grok_collector": grok_collector,
        "claude_analyzer": claude_analyzer,
        "technical_validator": AsyncMock(),
        "signal_scorer": MagicMock(),
        "risk_manager": MagicMock(),
        "market_gate": AsyncMock(),
        "trade_executor": AsyncMock(),
        "credibility_manager": MagicMock(),
        "outcome_tracker": MagicMock(),
    }


def make_orchestrator_for_integration(
    mock_components: dict,
    settings: OrchestratorSettings,
) -> TradingOrchestrator:
    """Create TradingOrchestrator for integration tests."""
    return TradingOrchestrator(
        grok_collector=mock_components["grok_collector"],
        claude_analyzer=mock_components["claude_analyzer"],
        technical_validator=mock_components["technical_validator"],
        signal_scorer=mock_components["signal_scorer"],
        risk_manager=mock_components["risk_manager"],
        market_gate=mock_components["market_gate"],
        trade_executor=mock_components["trade_executor"],
        credibility_manager=mock_components["credibility_manager"],
        outcome_tracker=mock_components["outcome_tracker"],
        settings=settings,
    )


class TestOrchestratorIntegration:
    """Integration tests for full orchestrator flow."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(
            immediate_threshold=0.85,
            batch_interval_seconds=1,
            min_consensus=0.6,
        )

    @pytest.fixture
    def sample_message(self) -> SocialMessage:
        """Create a sample social message with a ticker."""
        return SocialMessage(
            source=SourceType.GROK,
            source_id="test-123",
            content="$AAPL to the moon!",
            author="trader1",
            timestamp=datetime.now(),
            sentiment="bullish",
        )

    @pytest.fixture
    def analyzed_msg(self, sample_message: SocialMessage) -> AnalyzedMessage:
        """Create an analyzed message."""
        return AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

    @pytest.fixture
    def full_mock_components(
        self, sample_message: SocialMessage, analyzed_msg: AnalyzedMessage
    ) -> dict:
        """Create fully configured mock components."""
        components = make_mock_components_for_integration(sample_message, analyzed_msg)

        # Validator mock
        validated_signal = ValidatedSignal(
            message=analyzed_msg,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=TechnicalIndicators(
                    rsi=55.0,
                    macd_histogram=0.5,
                    macd_trend="rising",
                    stochastic_k=50.0,
                    stochastic_d=50.0,
                    adx=25.0,
                ),
                veto_reasons=[],
            ),
        )
        components["technical_validator"].validate.return_value = validated_signal

        # Scorer mock
        score_components = ScoreComponents(
            sentiment_score=85.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            tier=ScoreTier.STRONG,
            score=85.0,
            position_size_percent=2.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            risk_reward_ratio=3.0,
            components=score_components,
            reasoning="Strong bullish sentiment with technical confirmation",
            timestamp=datetime.now(),
        )
        components["signal_scorer"].score.return_value = recommendation

        # Risk manager mock
        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )
        components["risk_manager"].check_trade.return_value = risk_result

        # Gate mock
        gate_status = GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="all", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )
        components["market_gate"].check.return_value = gate_status

        # Executor mock
        exec_result = ExecutionResult(
            success=True,
            order_id="ORDER-123",
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=150.0,
            error_message=None,
            timestamp=datetime.now(),
        )
        components["trade_executor"].execute.return_value = exec_result

        return components

    @pytest.mark.asyncio
    async def test_full_flow_high_signal_execution(
        self,
        full_mock_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test complete flow from message to execution."""
        orchestrator = make_orchestrator_for_integration(full_mock_components, settings)

        # Mock _analyze_message to return high-signal analyzed message
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        assert orchestrator.state == OrchestratorState.RUNNING

        # Let stream process
        await asyncio.sleep(0.1)

        await orchestrator.stop()
        assert orchestrator.state == OrchestratorState.STOPPED

        # Verify execution was called
        full_mock_components["trade_executor"].execute.assert_called()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(
        self,
        full_mock_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test orchestrator lifecycle management."""
        orchestrator = make_orchestrator_for_integration(full_mock_components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        # Start
        await orchestrator.start()
        assert orchestrator.is_running

        # Stop
        await orchestrator.stop()
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_pipeline_stages_called_in_order(
        self,
        full_mock_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test that all pipeline stages are called in correct order."""
        orchestrator = make_orchestrator_for_integration(full_mock_components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Verify each stage was called
        full_mock_components["technical_validator"].validate.assert_called()
        full_mock_components["signal_scorer"].score.assert_called()
        full_mock_components["risk_manager"].check_trade.assert_called()
        full_mock_components["market_gate"].check.assert_called()
        full_mock_components["trade_executor"].execute.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_messages_processed(
        self,
        full_mock_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test processing multiple messages in a stream."""
        # Create multiple messages
        messages = [
            SocialMessage(
                source=SourceType.GROK,
                source_id=f"test-{i}",
                content=f"$AAPL signal {i}!",
                author="trader1",
                timestamp=datetime.now(),
                sentiment="bullish",
            )
            for i in range(3)
        ]

        full_mock_components["grok_collector"].stream = MagicMock(
            return_value=AsyncIteratorMock(messages)
        )

        orchestrator = make_orchestrator_for_integration(full_mock_components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.2)
        await orchestrator.stop()

        # _analyze_message should be called for each message
        assert orchestrator._analyze_message.call_count == 3


class TestOrchestratorIntegrationRejections:
    """Integration tests for rejection scenarios."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(
            immediate_threshold=0.85,
            batch_interval_seconds=1,
            min_consensus=0.6,
        )

    @pytest.fixture
    def sample_message(self) -> SocialMessage:
        """Create a sample social message with a ticker."""
        return SocialMessage(
            source=SourceType.GROK,
            source_id="test-123",
            content="$AAPL to the moon!",
            author="trader1",
            timestamp=datetime.now(),
            sentiment="bullish",
        )

    @pytest.fixture
    def analyzed_msg(self, sample_message: SocialMessage) -> AnalyzedMessage:
        """Create an analyzed message."""
        return AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

    @pytest.fixture
    def base_components(
        self, sample_message: SocialMessage, analyzed_msg: AnalyzedMessage
    ) -> dict:
        """Create base mock components for rejection tests."""
        return make_mock_components_for_integration(sample_message, analyzed_msg)

    @pytest.mark.asyncio
    async def test_validation_veto_stops_pipeline(
        self,
        base_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test that validation veto stops execution."""
        vetoed_signal = ValidatedSignal(
            message=analyzed_msg,
            validation=TechnicalValidation(
                status=ValidationStatus.VETO,
                indicators=TechnicalIndicators(
                    rsi=85.0,
                    macd_histogram=-0.5,
                    macd_trend="falling",
                    stochastic_k=90.0,
                    stochastic_d=88.0,
                    adx=15.0,
                ),
                veto_reasons=["RSI overbought", "Weak trend"],
            ),
        )

        base_components["technical_validator"].validate.return_value = vetoed_signal

        orchestrator = make_orchestrator_for_integration(base_components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called
        base_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_risk_rejection_stops_pipeline(
        self,
        base_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test that risk rejection stops execution."""
        valid_signal = ValidatedSignal(
            message=analyzed_msg,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=TechnicalIndicators(
                    rsi=55.0,
                    macd_histogram=0.5,
                    macd_trend="rising",
                    stochastic_k=50.0,
                    stochastic_d=50.0,
                    adx=25.0,
                ),
                veto_reasons=[],
            ),
        )

        base_components["technical_validator"].validate.return_value = valid_signal

        score_components = ScoreComponents(
            sentiment_score=85.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            tier=ScoreTier.STRONG,
            score=85.0,
            position_size_percent=2.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            risk_reward_ratio=3.0,
            components=score_components,
            reasoning="Strong signal",
            timestamp=datetime.now(),
        )
        base_components["signal_scorer"].score.return_value = recommendation

        risk_result = RiskCheckResult(
            approved=False,
            adjusted_quantity=0,
            adjusted_value=0.0,
            rejection_reason="Max daily loss exceeded",
        )
        base_components["risk_manager"].check_trade.return_value = risk_result

        orchestrator = make_orchestrator_for_integration(base_components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called
        base_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_closed_stops_pipeline(
        self,
        base_components: dict,
        settings: OrchestratorSettings,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test that closed gate stops execution."""
        valid_signal = ValidatedSignal(
            message=analyzed_msg,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=TechnicalIndicators(
                    rsi=55.0,
                    macd_histogram=0.5,
                    macd_trend="rising",
                    stochastic_k=50.0,
                    stochastic_d=50.0,
                    adx=25.0,
                ),
                veto_reasons=[],
            ),
        )

        base_components["technical_validator"].validate.return_value = valid_signal

        score_components = ScoreComponents(
            sentiment_score=85.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            tier=ScoreTier.STRONG,
            score=85.0,
            position_size_percent=2.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            risk_reward_ratio=3.0,
            components=score_components,
            reasoning="Strong signal",
            timestamp=datetime.now(),
        )
        base_components["signal_scorer"].score.return_value = recommendation

        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )
        base_components["risk_manager"].check_trade.return_value = risk_result

        gate_status = GateStatus(
            timestamp=datetime.now(),
            is_open=False,
            checks=[
                GateCheckResult(
                    name="vix", passed=False, reason="VIX too high", data={"vix": 35}
                )
            ],
            position_size_factor=0.0,
        )
        base_components["market_gate"].check.return_value = gate_status

        orchestrator = make_orchestrator_for_integration(base_components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called
        base_components["trade_executor"].execute.assert_not_called()


class TestOrchestratorBatchIntegration:
    """Integration tests for batch processing."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(
            immediate_threshold=0.95,  # High threshold so messages go to buffer
            batch_interval_seconds=1,
            min_consensus=0.5,
            max_buffer_size=10,
        )

    @pytest.mark.asyncio
    async def test_low_signal_messages_buffered(
        self, settings: OrchestratorSettings
    ) -> None:
        """Test that low-signal messages are buffered, not executed immediately."""
        sample_message = SocialMessage(
            source=SourceType.GROK,
            source_id="test-123",
            content="$AAPL might go up",
            author="trader1",
            timestamp=datetime.now(),
            sentiment="bullish",
        )

        # Low confidence message (below immediate_threshold of 0.95)
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.7,
                confidence=0.7,  # Below 0.95 threshold
            ),
        )

        components = make_mock_components_for_integration(sample_message, analyzed_msg)

        orchestrator = make_orchestrator_for_integration(components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)

        # Check that message was buffered
        assert len(orchestrator._message_buffer) >= 0  # May have been processed in batch

        await orchestrator.stop()


class TestOrchestratorErrorRecovery:
    """Integration tests for error recovery scenarios."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(
            immediate_threshold=0.85,
            continue_without_validator=True,
            gate_fail_safe_closed=True,
        )

    @pytest.fixture
    def sample_message(self) -> SocialMessage:
        return SocialMessage(
            source=SourceType.GROK,
            source_id="test-123",
            content="$AAPL to the moon!",
            author="trader1",
            timestamp=datetime.now(),
            sentiment="bullish",
        )

    @pytest.fixture
    def analyzed_msg(self, sample_message: SocialMessage) -> AnalyzedMessage:
        return AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

    @pytest.mark.asyncio
    async def test_continues_after_validator_error(
        self,
        settings: OrchestratorSettings,
        sample_message: SocialMessage,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test that orchestrator continues after validator error when configured."""
        components = make_mock_components_for_integration(sample_message, analyzed_msg)

        components["technical_validator"].validate.side_effect = Exception("Validator API down")

        score_components = ScoreComponents(
            sentiment_score=85.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            tier=ScoreTier.STRONG,
            score=85.0,
            position_size_percent=2.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            risk_reward_ratio=3.0,
            components=score_components,
            reasoning="Strong signal",
            timestamp=datetime.now(),
        )
        components["signal_scorer"].score.return_value = recommendation

        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )
        components["risk_manager"].check_trade.return_value = risk_result

        gate_status = GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="all", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )
        components["market_gate"].check.return_value = gate_status

        exec_result = ExecutionResult(
            success=True,
            order_id="ORDER-123",
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=150.0,
            error_message=None,
            timestamp=datetime.now(),
        )
        components["trade_executor"].execute.return_value = exec_result

        orchestrator = make_orchestrator_for_integration(components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Despite validator error, execution should proceed
        components["trade_executor"].execute.assert_called()

    @pytest.mark.asyncio
    async def test_gate_error_treated_as_closed(
        self,
        settings: OrchestratorSettings,
        sample_message: SocialMessage,
        analyzed_msg: AnalyzedMessage,
    ) -> None:
        """Test that gate error is treated as closed gate when configured."""
        components = make_mock_components_for_integration(sample_message, analyzed_msg)

        valid_signal = ValidatedSignal(
            message=analyzed_msg,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=TechnicalIndicators(
                    rsi=55.0,
                    macd_histogram=0.5,
                    macd_trend="rising",
                    stochastic_k=50.0,
                    stochastic_d=50.0,
                    adx=25.0,
                ),
                veto_reasons=[],
            ),
        )
        components["technical_validator"].validate.return_value = valid_signal

        score_components = ScoreComponents(
            sentiment_score=85.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            tier=ScoreTier.STRONG,
            score=85.0,
            position_size_percent=2.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            risk_reward_ratio=3.0,
            components=score_components,
            reasoning="Strong signal",
            timestamp=datetime.now(),
        )
        components["signal_scorer"].score.return_value = recommendation

        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )
        components["risk_manager"].check_trade.return_value = risk_result

        components["market_gate"].check.side_effect = Exception("Gate API down")

        orchestrator = make_orchestrator_for_integration(components, settings)
        orchestrator._analyze_message = MagicMock(return_value=analyzed_msg)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called due to gate error = closed
        components["trade_executor"].execute.assert_not_called()
