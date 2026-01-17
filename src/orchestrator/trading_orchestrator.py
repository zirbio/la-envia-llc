"""Main trading orchestrator that coordinates all pipeline components."""

import asyncio
import logging

from analyzers.analyzed_message import AnalyzedMessage
from analyzers.analyzer_manager import AnalyzerManager
from collectors.collector_manager import CollectorManager
from execution.trade_executor import TradeExecutor
from gate.market_gate import MarketGate
from models.social_message import SocialMessage
from orchestrator.models import OrchestratorState, ProcessResult
from orchestrator.settings import OrchestratorSettings
from risk.risk_manager import RiskManager
from scoring.models import ScoreTier
from scoring.signal_scorer import SignalScorer
from validators.models import TechnicalIndicators, TechnicalValidation, ValidatedSignal, ValidationStatus
from validators.technical_validator import TechnicalValidator


logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Coordinates all trading pipeline components.

    Implements hybrid processing:
    - High-signal messages (confidence >= threshold) processed immediately
    - Regular messages batched and processed periodically
    """

    def __init__(
        self,
        collector_manager: CollectorManager,
        analyzer_manager: AnalyzerManager,
        technical_validator: TechnicalValidator,
        signal_scorer: SignalScorer,
        risk_manager: RiskManager,
        market_gate: MarketGate,
        trade_executor: TradeExecutor,
        settings: OrchestratorSettings,
    ):
        self._collector_manager = collector_manager
        self._analyzer = analyzer_manager
        self._validator = technical_validator
        self._scorer = signal_scorer
        self._risk_manager = risk_manager
        self._gate = market_gate
        self._executor = trade_executor
        self._settings = settings

        self._state = OrchestratorState.STOPPED
        self._message_buffer: list[AnalyzedMessage] = []
        self._stream_task: asyncio.Task | None = None
        self._batch_task: asyncio.Task | None = None

    @property
    def state(self) -> OrchestratorState:
        """Return the current orchestrator state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Return True if the orchestrator is in RUNNING state."""
        return self._state == OrchestratorState.RUNNING

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._state != OrchestratorState.STOPPED:
            raise RuntimeError("Orchestrator already running")

        self._state = OrchestratorState.RUNNING
        logger.info("Starting trading orchestrator")

        # Connect collectors
        await self._collector_manager.connect_all()

        # Register message callback
        self._collector_manager.add_callback(self._on_message_callback)

        # Start background tasks
        self._stream_task = asyncio.create_task(self._run_stream())
        self._batch_task = asyncio.create_task(self._batch_processor())

        logger.info("Trading orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        if self._state == OrchestratorState.STOPPED:
            return

        self._state = OrchestratorState.STOPPING
        logger.info("Stopping trading orchestrator")

        # Cancel tasks
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Process remaining buffer
        if self._message_buffer:
            logger.info(f"Processing {len(self._message_buffer)} buffered messages")
            await self._flush_buffer()

        # Disconnect collectors
        await self._collector_manager.disconnect_all()

        self._state = OrchestratorState.STOPPED
        logger.info("Trading orchestrator stopped")

    async def _run_stream(self) -> None:
        """Main stream loop."""
        try:
            async for message in self._collector_manager.stream_all():
                if self._state != OrchestratorState.RUNNING:
                    break
                await self._on_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Stream error: {e}")

    async def _batch_processor(self) -> None:
        """Background task for batch processing."""
        try:
            while self._state == OrchestratorState.RUNNING:
                await asyncio.sleep(self._settings.batch_interval_seconds)
                if self._message_buffer:
                    await self._process_batch()
        except asyncio.CancelledError:
            pass

    async def _flush_buffer(self) -> None:
        """Process all remaining messages in buffer."""
        if self._message_buffer:
            await self._process_batch()

    async def _process_batch(self) -> None:
        """Process current batch buffer (placeholder)."""
        self._message_buffer.clear()

    def _on_message_callback(self, message: SocialMessage) -> None:
        """Sync callback wrapper for collector manager."""
        asyncio.create_task(self._on_message(message))

    async def _on_message(self, message: SocialMessage) -> ProcessResult:
        """Process incoming message from collectors.

        Routes high-signal messages to immediate processing,
        others to batch buffer.

        Args:
            message: Raw social message from collector.

        Returns:
            ProcessResult indicating outcome.
        """
        try:
            # Step 1: Analyze message
            analyzed = await self._analyzer.analyze(message)

            # Step 2: Route based on signal strength
            if self._is_high_signal(analyzed):
                return await self._process_immediate(analyzed)
            else:
                self._message_buffer.append(analyzed)
                # Check buffer size limit
                if len(self._message_buffer) >= self._settings.max_buffer_size:
                    await self._process_batch()
                return ProcessResult(status="buffered")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ProcessResult(status="error", error=str(e))

    async def _process_immediate(self, analyzed: AnalyzedMessage) -> ProcessResult:
        """Process high-signal message through full pipeline.

        Pipeline: validate -> score -> risk check -> gate check -> execute

        Args:
            analyzed: Analyzed message from sentiment analysis.

        Returns:
            ProcessResult with status and details of processing outcome.
        """
        symbol = self._extract_symbol(analyzed)

        # Step 1: Technical validation
        try:
            validated = await self._validator.validate(analyzed)
        except Exception as e:
            if self._settings.continue_without_validator:
                logger.warning(f"Validator error, continuing: {e}")
                validated = self._create_unvalidated_signal(analyzed)
            else:
                return ProcessResult(status="error", symbol=symbol, error=f"Validation error: {e}")

        if not validated.should_trade():
            return ProcessResult(status="vetoed", symbol=symbol)

        # Step 2: Scoring
        try:
            recommendation = self._scorer.score(validated)
        except Exception as e:
            return ProcessResult(status="error", symbol=symbol, error=f"Scoring error: {e}")

        if recommendation.tier == ScoreTier.NO_TRADE:
            return ProcessResult(status="low_score", symbol=symbol, recommendation=recommendation)

        # Step 3: Risk check
        try:
            risk_result = self._risk_manager.check_trade(
                recommendation,
                recommendation.position_size_percent,
                self._get_current_price(symbol),
            )
        except Exception as e:
            return ProcessResult(status="error", symbol=symbol, error=f"Risk error: {e}")

        if not risk_result.approved:
            return ProcessResult(status="risk_rejected", symbol=symbol, recommendation=recommendation)

        # Step 4: Gate check
        try:
            gate_status = await self._gate.check()
        except Exception as e:
            if self._settings.gate_fail_safe_closed:
                logger.warning(f"Gate error, treating as closed: {e}")
                return ProcessResult(status="gate_closed", symbol=symbol, recommendation=recommendation)
            else:
                return ProcessResult(status="error", symbol=symbol, error=f"Gate error: {e}")

        if not gate_status.is_open:
            return ProcessResult(status="gate_closed", symbol=symbol, recommendation=recommendation)

        # Step 5: Execute
        try:
            exec_result = await self._executor.execute(recommendation, risk_result, gate_status)
        except Exception as e:
            return ProcessResult(status="error", symbol=symbol, error=f"Execution error: {e}")

        return ProcessResult(
            status="executed",
            symbol=symbol,
            recommendation=recommendation,
            execution_result=exec_result,
        )

    def _extract_symbol(self, analyzed: AnalyzedMessage) -> str | None:
        """Extract primary symbol from analyzed message.

        Args:
            analyzed: Analyzed message to extract symbol from.

        Returns:
            First ticker symbol found, or None if no tickers.
        """
        tickers = analyzed.get_tickers()
        return tickers[0] if tickers else None

    def _get_current_price(self, symbol: str | None) -> float:
        """Get current price for symbol.

        Note: This is a placeholder. In production, this would
        fetch real-time price from a market data source.

        Args:
            symbol: Ticker symbol to get price for.

        Returns:
            Current price (placeholder returns 150.0).
        """
        return 150.0

    def _create_unvalidated_signal(self, analyzed: AnalyzedMessage) -> ValidatedSignal:
        """Create pass-through signal when validator fails.

        Used when continue_without_validator is True and the
        technical validator encounters an error.

        Args:
            analyzed: Original analyzed message.

        Returns:
            ValidatedSignal with PASS status and warning.
        """
        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=TechnicalIndicators(
                rsi=50.0,
                macd_histogram=0.0,
                macd_trend="flat",
                stochastic_k=50.0,
                stochastic_d=50.0,
                adx=25.0,
            ),
            veto_reasons=[],
            warnings=["Validation skipped due to error"],
        )
        return ValidatedSignal(message=analyzed, validation=validation)

    def _is_high_signal(self, msg: AnalyzedMessage) -> bool:
        """Determine if message requires immediate processing.

        Args:
            msg: Analyzed message to check.

        Returns:
            True if high-signal (high confidence and not neutral).
        """
        return (
            msg.sentiment.confidence >= self._settings.immediate_threshold
            and msg.sentiment.label != "neutral"
        )
