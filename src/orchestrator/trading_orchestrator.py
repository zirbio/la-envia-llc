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
from scoring.signal_scorer import SignalScorer
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
        """Process high-signal message immediately (placeholder)."""
        return ProcessResult(status="executed")

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
