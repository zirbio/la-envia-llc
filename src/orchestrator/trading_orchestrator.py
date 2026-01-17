"""Main trading orchestrator that coordinates all pipeline components."""

import asyncio
import logging

from analyzers.analyzed_message import AnalyzedMessage
from analyzers.analyzer_manager import AnalyzerManager
from collectors.collector_manager import CollectorManager
from execution.trade_executor import TradeExecutor
from gate.market_gate import MarketGate
from orchestrator.models import OrchestratorState
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
