"""Data models for trading orchestrator."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from execution.models import ExecutionResult
from scoring.models import TradeRecommendation


class OrchestratorState(Enum):
    """State of the trading orchestrator."""

    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass
class ProcessResult:
    """Result of processing a message through the pipeline."""

    status: str
    symbol: str | None = None
    recommendation: TradeRecommendation | None = None
    execution_result: ExecutionResult | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple messages for a symbol."""

    symbol: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_count: int
    consensus_label: str
    consensus_strength: float
    avg_confidence: float
