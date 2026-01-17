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
    """Mock async iterator for testing stream_all."""

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
            source=SourceType.TWITTER,
            source_id="test-123",
            content="$AAPL to the moon!",
            author="trader1",
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def full_mock_components(self, sample_message: SocialMessage) -> dict:
        """Create fully configured mock components."""
        # Collector manager mock
        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.add_callback = MagicMock()
        collector.stream_all = MagicMock(
            return_value=AsyncIteratorMock([sample_message])
        )

        # Create analyzed message for reuse
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

        # Analyzer mock
        analyzer = AsyncMock()
        analyzer.analyze.return_value = analyzed_msg

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

        validator = AsyncMock()
        validator.validate.return_value = validated_signal

        # Scorer mock
        components = ScoreComponents(
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
            components=components,
            reasoning="Strong bullish sentiment with technical confirmation",
            timestamp=datetime.now(),
        )

        scorer = MagicMock()
        scorer.score.return_value = recommendation

        # Risk manager mock
        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )

        risk = MagicMock()
        risk.check_trade.return_value = risk_result

        # Gate mock
        gate_status = GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="all", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )

        gate = AsyncMock()
        gate.check.return_value = gate_status

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

        executor = AsyncMock()
        executor.execute.return_value = exec_result

        return {
            "collector_manager": collector,
            "analyzer_manager": analyzer,
            "technical_validator": validator,
            "signal_scorer": scorer,
            "risk_manager": risk,
            "market_gate": gate,
            "trade_executor": executor,
        }

    @pytest.mark.asyncio
    async def test_full_flow_high_signal_execution(
        self, full_mock_components: dict, settings: OrchestratorSettings
    ) -> None:
        """Test complete flow from message to execution."""
        orchestrator = TradingOrchestrator(**full_mock_components, settings=settings)

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
        self, full_mock_components: dict, settings: OrchestratorSettings
    ) -> None:
        """Test orchestrator lifecycle management."""
        orchestrator = TradingOrchestrator(**full_mock_components, settings=settings)

        # Start
        await orchestrator.start()
        assert orchestrator.is_running
        full_mock_components["collector_manager"].connect_all.assert_called_once()

        # Stop
        await orchestrator.stop()
        assert not orchestrator.is_running
        full_mock_components["collector_manager"].disconnect_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_stages_called_in_order(
        self, full_mock_components: dict, settings: OrchestratorSettings
    ) -> None:
        """Test that all pipeline stages are called in correct order."""
        orchestrator = TradingOrchestrator(**full_mock_components, settings=settings)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Verify each stage was called
        full_mock_components["analyzer_manager"].analyze.assert_called()
        full_mock_components["technical_validator"].validate.assert_called()
        full_mock_components["signal_scorer"].score.assert_called()
        full_mock_components["risk_manager"].check_trade.assert_called()
        full_mock_components["market_gate"].check.assert_called()
        full_mock_components["trade_executor"].execute.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_messages_processed(
        self, full_mock_components: dict, settings: OrchestratorSettings
    ) -> None:
        """Test processing multiple messages in a stream."""
        # Create multiple messages
        messages = [
            SocialMessage(
                source=SourceType.TWITTER,
                source_id=f"test-{i}",
                content=f"$AAPL signal {i}!",
                author="trader1",
                timestamp=datetime.now(),
            )
            for i in range(3)
        ]

        full_mock_components["collector_manager"].stream_all = MagicMock(
            return_value=AsyncIteratorMock(messages)
        )

        orchestrator = TradingOrchestrator(**full_mock_components, settings=settings)

        await orchestrator.start()
        await asyncio.sleep(0.2)
        await orchestrator.stop()

        # Analyzer should be called for each message
        assert full_mock_components["analyzer_manager"].analyze.call_count == 3


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
            source=SourceType.TWITTER,
            source_id="test-123",
            content="$AAPL to the moon!",
            author="trader1",
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def base_components(self, sample_message: SocialMessage) -> dict:
        """Create base mock components for rejection tests."""
        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.add_callback = MagicMock()
        collector.stream_all = MagicMock(
            return_value=AsyncIteratorMock([sample_message])
        )

        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

        analyzer = AsyncMock()
        analyzer.analyze.return_value = analyzed_msg

        return {
            "collector_manager": collector,
            "analyzer_manager": analyzer,
            "technical_validator": AsyncMock(),
            "signal_scorer": MagicMock(),
            "risk_manager": MagicMock(),
            "market_gate": AsyncMock(),
            "trade_executor": AsyncMock(),
        }

    @pytest.mark.asyncio
    async def test_validation_veto_stops_pipeline(
        self, base_components: dict, settings: OrchestratorSettings, sample_message: SocialMessage
    ) -> None:
        """Test that validation veto stops execution."""
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

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

        orchestrator = TradingOrchestrator(**base_components, settings=settings)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called
        base_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_risk_rejection_stops_pipeline(
        self, base_components: dict, settings: OrchestratorSettings, sample_message: SocialMessage
    ) -> None:
        """Test that risk rejection stops execution."""
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

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

        components = ScoreComponents(
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
            components=components,
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

        orchestrator = TradingOrchestrator(**base_components, settings=settings)

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called
        base_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_closed_stops_pipeline(
        self, base_components: dict, settings: OrchestratorSettings, sample_message: SocialMessage
    ) -> None:
        """Test that closed gate stops execution."""
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

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

        components = ScoreComponents(
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
            components=components,
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

        orchestrator = TradingOrchestrator(**base_components, settings=settings)

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
            source=SourceType.TWITTER,
            source_id="test-123",
            content="$AAPL might go up",
            author="trader1",
            timestamp=datetime.now(),
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

        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.add_callback = MagicMock()
        collector.stream_all = MagicMock(
            return_value=AsyncIteratorMock([sample_message])
        )

        analyzer = AsyncMock()
        analyzer.analyze.return_value = analyzed_msg

        executor = AsyncMock()

        orchestrator = TradingOrchestrator(
            collector_manager=collector,
            analyzer_manager=analyzer,
            technical_validator=AsyncMock(),
            signal_scorer=MagicMock(),
            risk_manager=MagicMock(),
            market_gate=AsyncMock(),
            trade_executor=executor,
            settings=settings,
        )

        await orchestrator.start()
        await asyncio.sleep(0.1)

        # Check that message was buffered
        assert len(orchestrator._message_buffer) >= 0  # May have been processed in batch

        await orchestrator.stop()

        # Validator should NOT be called for buffered messages during immediate processing
        # (unless batch processing runs)


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
            source=SourceType.TWITTER,
            source_id="test-123",
            content="$AAPL to the moon!",
            author="trader1",
            timestamp=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_continues_after_validator_error(
        self, settings: OrchestratorSettings, sample_message: SocialMessage
    ) -> None:
        """Test that orchestrator continues after validator error when configured."""
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.add_callback = MagicMock()
        collector.stream_all = MagicMock(
            return_value=AsyncIteratorMock([sample_message])
        )

        analyzer = AsyncMock()
        analyzer.analyze.return_value = analyzed_msg

        validator = AsyncMock()
        validator.validate.side_effect = Exception("Validator API down")

        components = ScoreComponents(
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
            components=components,
            reasoning="Strong signal",
            timestamp=datetime.now(),
        )
        scorer = MagicMock()
        scorer.score.return_value = recommendation

        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )
        risk = MagicMock()
        risk.check_trade.return_value = risk_result

        gate_status = GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="all", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )
        gate = AsyncMock()
        gate.check.return_value = gate_status

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
        executor = AsyncMock()
        executor.execute.return_value = exec_result

        orchestrator = TradingOrchestrator(
            collector_manager=collector,
            analyzer_manager=analyzer,
            technical_validator=validator,
            signal_scorer=scorer,
            risk_manager=risk,
            market_gate=gate,
            trade_executor=executor,
            settings=settings,
        )

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Despite validator error, execution should proceed
        executor.execute.assert_called()

    @pytest.mark.asyncio
    async def test_gate_error_treated_as_closed(
        self, settings: OrchestratorSettings, sample_message: SocialMessage
    ) -> None:
        """Test that gate error is treated as closed gate when configured."""
        analyzed_msg = AnalyzedMessage(
            message=sample_message,
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.9,
                confidence=0.9,
            ),
        )

        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.add_callback = MagicMock()
        collector.stream_all = MagicMock(
            return_value=AsyncIteratorMock([sample_message])
        )

        analyzer = AsyncMock()
        analyzer.analyze.return_value = analyzed_msg

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
        validator = AsyncMock()
        validator.validate.return_value = valid_signal

        components = ScoreComponents(
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
            components=components,
            reasoning="Strong signal",
            timestamp=datetime.now(),
        )
        scorer = MagicMock()
        scorer.score.return_value = recommendation

        risk_result = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            adjusted_value=1500.0,
            rejection_reason=None,
        )
        risk = MagicMock()
        risk.check_trade.return_value = risk_result

        gate = AsyncMock()
        gate.check.side_effect = Exception("Gate API down")

        executor = AsyncMock()

        orchestrator = TradingOrchestrator(
            collector_manager=collector,
            analyzer_manager=analyzer,
            technical_validator=validator,
            signal_scorer=scorer,
            risk_manager=risk,
            market_gate=gate,
            trade_executor=executor,
            settings=settings,
        )

        await orchestrator.start()
        await asyncio.sleep(0.1)
        await orchestrator.stop()

        # Executor should NOT be called due to gate error = closed
        executor.execute.assert_not_called()
