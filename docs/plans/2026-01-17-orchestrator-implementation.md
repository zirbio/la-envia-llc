# Phase 8: Trading Orchestrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement TradingOrchestrator that coordinates all pipeline components with hybrid processing mode.

**Architecture:** Hybrid processing (immediate for high-signal, batch for regular), graceful degradation, manual start/stop control.

**Tech Stack:** Python asyncio, dataclasses, Pydantic settings, enum

---

## Task 1: Data Models (OrchestratorState, ProcessResult)

**Files:**
- Create: `src/orchestrator/__init__.py`
- Create: `src/orchestrator/models.py`
- Create: `tests/orchestrator/__init__.py`
- Create: `tests/orchestrator/test_models.py`

**Step 1: Create test file**

```python
# tests/orchestrator/test_models.py
"""Tests for orchestrator data models."""

from datetime import datetime

import pytest

from orchestrator.models import AggregatedSentiment, OrchestratorState, ProcessResult


class TestOrchestratorState:
    """Tests for OrchestratorState enum."""

    def test_has_stopped_state(self) -> None:
        """OrchestratorState has STOPPED value."""
        assert OrchestratorState.STOPPED.value == "stopped"

    def test_has_running_state(self) -> None:
        """OrchestratorState has RUNNING value."""
        assert OrchestratorState.RUNNING.value == "running"

    def test_has_stopping_state(self) -> None:
        """OrchestratorState has STOPPING value."""
        assert OrchestratorState.STOPPING.value == "stopping"


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_create_executed_result(self) -> None:
        """ProcessResult can be created with executed status."""
        result = ProcessResult(
            status="executed",
            symbol="AAPL",
        )
        assert result.status == "executed"
        assert result.symbol == "AAPL"
        assert result.error is None

    def test_create_error_result(self) -> None:
        """ProcessResult can be created with error status."""
        result = ProcessResult(
            status="error",
            error="Scoring failed",
        )
        assert result.status == "error"
        assert result.error == "Scoring failed"

    def test_create_vetoed_result(self) -> None:
        """ProcessResult can be created with vetoed status."""
        result = ProcessResult(
            status="vetoed",
            symbol="TSLA",
        )
        assert result.status == "vetoed"

    def test_timestamp_auto_set(self) -> None:
        """ProcessResult sets timestamp automatically."""
        result = ProcessResult(status="buffered")
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestAggregatedSentiment:
    """Tests for AggregatedSentiment dataclass."""

    def test_create_aggregated_sentiment(self) -> None:
        """AggregatedSentiment can be created."""
        agg = AggregatedSentiment(
            symbol="AAPL",
            bullish_count=5,
            bearish_count=2,
            neutral_count=1,
            total_count=8,
            consensus_label="bullish",
            consensus_strength=0.625,
            avg_confidence=0.75,
        )
        assert agg.symbol == "AAPL"
        assert agg.consensus_label == "bullish"
        assert agg.consensus_strength == 0.625

    def test_consensus_strength_calculation(self) -> None:
        """Consensus strength is dominant / total."""
        agg = AggregatedSentiment(
            symbol="TSLA",
            bullish_count=8,
            bearish_count=2,
            neutral_count=0,
            total_count=10,
            consensus_label="bullish",
            consensus_strength=0.8,  # 8/10
            avg_confidence=0.85,
        )
        assert agg.consensus_strength == 0.8
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'orchestrator'"

**Step 3: Create the models**

```python
# src/orchestrator/__init__.py
"""Orchestrator module for coordinating trading pipeline."""

from .models import AggregatedSentiment, OrchestratorState, ProcessResult

__all__ = ["AggregatedSentiment", "OrchestratorState", "ProcessResult"]
```

```python
# src/orchestrator/models.py
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
    """Result of processing a message through the pipeline.

    Attributes:
        status: Processing outcome (executed, vetoed, low_score, gate_closed, error, buffered).
        symbol: Stock symbol if applicable.
        recommendation: Trade recommendation if generated.
        execution_result: Execution result if trade was executed.
        error: Error message if processing failed.
        timestamp: When processing completed.
    """

    status: str
    symbol: str | None = None
    recommendation: TradeRecommendation | None = None
    execution_result: ExecutionResult | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple messages for a symbol.

    Attributes:
        symbol: Stock symbol.
        bullish_count: Number of bullish messages.
        bearish_count: Number of bearish messages.
        neutral_count: Number of neutral messages.
        total_count: Total messages.
        consensus_label: Dominant sentiment label.
        consensus_strength: Ratio of dominant sentiment to total.
        avg_confidence: Average confidence across messages.
    """

    symbol: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_count: int
    consensus_label: str
    consensus_strength: float
    avg_confidence: float
```

```python
# tests/orchestrator/__init__.py
"""Tests for orchestrator module."""
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_models.py -v
```

Expected: All 7 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add OrchestratorState and ProcessResult models"
```

---

## Task 2: OrchestratorSettings Configuration

**Files:**
- Create: `src/orchestrator/settings.py`
- Modify: `src/orchestrator/__init__.py`
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Create: `tests/config/test_orchestrator_settings.py`

**Step 1: Create settings test**

```python
# tests/config/test_orchestrator_settings.py
"""Tests for OrchestratorSettings configuration."""

import pytest

from config.settings import Settings
from orchestrator.settings import OrchestratorSettings


class TestOrchestratorSettings:
    """Tests for orchestrator settings."""

    def test_default_values(self) -> None:
        """OrchestratorSettings has correct defaults."""
        settings = OrchestratorSettings()

        assert settings.enabled is True
        assert settings.immediate_threshold == 0.85
        assert settings.batch_interval_seconds == 60
        assert settings.min_consensus == 0.6
        assert settings.max_buffer_size == 1000
        assert settings.continue_without_validator is True
        assert settings.gate_fail_safe_closed is True

    def test_custom_values(self) -> None:
        """OrchestratorSettings accepts custom values."""
        settings = OrchestratorSettings(
            immediate_threshold=0.9,
            batch_interval_seconds=30,
        )
        assert settings.immediate_threshold == 0.9
        assert settings.batch_interval_seconds == 30

    def test_settings_has_orchestrator_config(self) -> None:
        """Main Settings includes orchestrator config."""
        settings = Settings()
        assert hasattr(settings, "orchestrator")
        assert isinstance(settings.orchestrator, OrchestratorSettings)
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/config/test_orchestrator_settings.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create OrchestratorSettings**

```python
# src/orchestrator/settings.py
"""Configuration for trading orchestrator."""

from pydantic import BaseModel, Field


class OrchestratorSettings(BaseModel):
    """Settings for TradingOrchestrator.

    Attributes:
        enabled: Whether orchestrator is enabled.
        immediate_threshold: Confidence threshold for immediate processing.
        batch_interval_seconds: How often to process batch buffer.
        min_consensus: Minimum consensus strength to act on batch.
        max_buffer_size: Maximum messages in buffer before forced processing.
        continue_without_validator: Continue if validator fails.
        gate_fail_safe_closed: Treat gate as closed on error.
    """

    enabled: bool = True
    immediate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    batch_interval_seconds: int = Field(default=60, ge=1)
    min_consensus: float = Field(default=0.6, ge=0.0, le=1.0)
    max_buffer_size: int = Field(default=1000, ge=1)
    continue_without_validator: bool = True
    gate_fail_safe_closed: bool = True
```

Update `src/orchestrator/__init__.py`:

```python
# src/orchestrator/__init__.py
"""Orchestrator module for coordinating trading pipeline."""

from .models import AggregatedSentiment, OrchestratorState, ProcessResult
from .settings import OrchestratorSettings

__all__ = [
    "AggregatedSentiment",
    "OrchestratorState",
    "OrchestratorSettings",
    "ProcessResult",
]
```

Update `src/config/settings.py` - add import and field:

```python
# Add to imports
from orchestrator.settings import OrchestratorSettings

# Add to Settings class
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
```

Update `config/settings.yaml`:

```yaml
# Add after market_gate section

# Phase 8: Trading Orchestrator
orchestrator:
  enabled: true
  immediate_threshold: 0.85
  batch_interval_seconds: 60
  min_consensus: 0.6
  max_buffer_size: 1000
  continue_without_validator: true
  gate_fail_safe_closed: true
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/config/test_orchestrator_settings.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ src/config/ config/settings.yaml tests/config/ && git commit -m "feat(orchestrator): add OrchestratorSettings configuration"
```

---

## Task 3: TradingOrchestrator - Core Structure

**Files:**
- Create: `src/orchestrator/trading_orchestrator.py`
- Create: `tests/orchestrator/test_orchestrator.py`
- Modify: `src/orchestrator/__init__.py`

**Step 1: Create test file with core tests**

```python
# tests/orchestrator/test_orchestrator.py
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
        """Default settings for tests."""
        return OrchestratorSettings()

    @pytest.fixture
    def mock_components(self) -> dict:
        """Create mock components."""
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
        """Create orchestrator with mock components."""
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
        """Orchestrator starts in STOPPED state."""
        assert orchestrator.state == OrchestratorState.STOPPED

    def test_is_running_false_when_stopped(self, orchestrator: TradingOrchestrator) -> None:
        """is_running returns False when stopped."""
        assert orchestrator.is_running is False

    def test_buffer_initially_empty(self, orchestrator: TradingOrchestrator) -> None:
        """Message buffer starts empty."""
        assert len(orchestrator._message_buffer) == 0
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestTradingOrchestratorCore -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create TradingOrchestrator core**

```python
# src/orchestrator/trading_orchestrator.py
"""Main trading orchestrator that coordinates all pipeline components."""

import asyncio
import logging
from datetime import datetime

from analyzers.analyzed_message import AnalyzedMessage
from analyzers.analyzer_manager import AnalyzerManager
from collectors.collector_manager import CollectorManager
from execution.trade_executor import TradeExecutor
from gate.market_gate import MarketGate
from models.social_message import SocialMessage
from orchestrator.models import AggregatedSentiment, OrchestratorState, ProcessResult
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

    Attributes:
        state: Current orchestrator state.
        is_running: Whether orchestrator is actively processing.
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
        """Initialize TradingOrchestrator.

        Args:
            collector_manager: Manages social media collectors.
            analyzer_manager: Handles sentiment analysis.
            technical_validator: Validates signals technically.
            signal_scorer: Scores and generates recommendations.
            risk_manager: Manages risk and position sizing.
            market_gate: Checks market conditions.
            trade_executor: Executes trades.
            settings: Orchestrator configuration.
        """
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
        """Get current orchestrator state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._state == OrchestratorState.RUNNING
```

Update `src/orchestrator/__init__.py`:

```python
# src/orchestrator/__init__.py
"""Orchestrator module for coordinating trading pipeline."""

from .models import AggregatedSentiment, OrchestratorState, ProcessResult
from .settings import OrchestratorSettings
from .trading_orchestrator import TradingOrchestrator

__all__ = [
    "AggregatedSentiment",
    "OrchestratorState",
    "OrchestratorSettings",
    "ProcessResult",
    "TradingOrchestrator",
]
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestTradingOrchestratorCore -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add TradingOrchestrator core structure"
```

---

## Task 4: Start/Stop Methods

**Files:**
- Modify: `src/orchestrator/trading_orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

**Step 1: Add start/stop tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
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
        collector.stream_all = AsyncMock(return_value=AsyncIteratorMock([]))

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
        """start() changes state to RUNNING."""
        await orchestrator.start()
        assert orchestrator.state == OrchestratorState.RUNNING
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_start_connects_collectors(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """start() calls collector_manager.connect_all()."""
        await orchestrator.start()
        mock_components["collector_manager"].connect_all.assert_called_once()
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stop_changes_state_to_stopped(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """stop() changes state to STOPPED."""
        await orchestrator.start()
        await orchestrator.stop()
        assert orchestrator.state == OrchestratorState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_disconnects_collectors(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """stop() calls collector_manager.disconnect_all()."""
        await orchestrator.start()
        await orchestrator.stop()
        mock_components["collector_manager"].disconnect_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_raises_if_already_running(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """start() raises error if already running."""
        await orchestrator.start()
        with pytest.raises(RuntimeError, match="already running"):
            await orchestrator.start()
        await orchestrator.stop()


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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestTradingOrchestratorStartStop -v
```

Expected: FAIL with "AttributeError: 'TradingOrchestrator' object has no attribute 'start'"

**Step 3: Add start/stop methods**

Add to `src/orchestrator/trading_orchestrator.py`:

```python
    async def start(self) -> None:
        """Start the orchestrator.

        Connects to collectors, starts stream and batch tasks.

        Raises:
            RuntimeError: If already running.
        """
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
        """Stop the orchestrator gracefully.

        Processes remaining buffer, cancels tasks, disconnects collectors.
        """
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
        """Main stream loop - processes messages from collectors."""
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
        """Background task that processes batch buffer periodically."""
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
        """Process current batch buffer."""
        # Placeholder - will be implemented in Task 6
        self._message_buffer.clear()

    def _on_message_callback(self, message: SocialMessage) -> None:
        """Sync callback wrapper for collector manager."""
        asyncio.create_task(self._on_message(message))

    async def _on_message(self, message: SocialMessage) -> ProcessResult:
        """Process incoming message."""
        # Placeholder - will be implemented in Task 5
        return ProcessResult(status="buffered")
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestTradingOrchestratorStartStop -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add start/stop methods"
```

---

## Task 5: High-Signal Detection and Immediate Processing

**Files:**
- Modify: `src/orchestrator/trading_orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

**Step 1: Add high-signal detection tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
from analyzers.analyzed_message import AnalyzedMessage
from analyzers.sentiment_result import SentimentResult
from models.social_message import SocialMessage, SourceType


def make_social_message(content: str = "Test message") -> SocialMessage:
    """Create a test social message."""
    return SocialMessage(
        source=SourceType.TWITTER,
        content=content,
        author="test_user",
        timestamp=datetime.now(),
        raw_data={},
    )


def make_analyzed_message(
    confidence: float = 0.9,
    label: str = "bullish",
) -> AnalyzedMessage:
    """Create a test analyzed message."""
    sentiment = SentimentResult(
        label=label,
        confidence=confidence,
        scores={"bullish": confidence if label == "bullish" else 0.1},
    )
    return AnalyzedMessage(
        original=make_social_message(),
        sentiment_result=sentiment,
    )


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
        """_is_high_signal returns True for high confidence non-neutral."""
        msg = make_analyzed_message(confidence=0.9, label="bullish")
        assert orchestrator._is_high_signal(msg) is True

    def test_is_high_signal_false_for_low_confidence(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        """_is_high_signal returns False for low confidence."""
        msg = make_analyzed_message(confidence=0.7, label="bullish")
        assert orchestrator._is_high_signal(msg) is False

    def test_is_high_signal_false_for_neutral(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        """_is_high_signal returns False for neutral sentiment."""
        msg = make_analyzed_message(confidence=0.95, label="neutral")
        assert orchestrator._is_high_signal(msg) is False

    def test_is_high_signal_at_threshold(
        self, orchestrator: TradingOrchestrator
    ) -> None:
        """_is_high_signal returns True at exact threshold."""
        msg = make_analyzed_message(confidence=0.85, label="bearish")
        assert orchestrator._is_high_signal(msg) is True
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestHighSignalDetection -v
```

**Step 3: Add _is_high_signal method**

Add to `src/orchestrator/trading_orchestrator.py`:

```python
    def _is_high_signal(self, msg: AnalyzedMessage) -> bool:
        """Determine if message requires immediate processing.

        Args:
            msg: Analyzed message to check.

        Returns:
            True if high-signal (high confidence and not neutral).
        """
        return (
            msg.sentiment_result.confidence >= self._settings.immediate_threshold
            and msg.sentiment_result.label != "neutral"
        )
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestHighSignalDetection -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add high-signal detection"
```

---

## Task 6: Message Routing (_on_message)

**Files:**
- Modify: `src/orchestrator/trading_orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

**Step 1: Add message routing tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
class TestMessageRouting:
    """Tests for message routing logic."""

    @pytest.fixture
    def settings(self) -> OrchestratorSettings:
        return OrchestratorSettings(immediate_threshold=0.85)

    @pytest.fixture
    def mock_analyzer(self) -> AsyncMock:
        analyzer = AsyncMock()
        return analyzer

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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestMessageRouting -v
```

**Step 3: Implement _on_message**

Update `_on_message` in `src/orchestrator/trading_orchestrator.py`:

```python
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
        """Process high-signal message immediately through full pipeline.

        Args:
            analyzed: Analyzed message to process.

        Returns:
            ProcessResult with outcome.
        """
        # Placeholder - will be implemented in Task 7
        return ProcessResult(status="executed")
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestMessageRouting -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add message routing logic"
```

---

## Task 7: Immediate Processing Pipeline

**Files:**
- Modify: `src/orchestrator/trading_orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

**Step 1: Add immediate processing tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
from gate.models import GateCheckResult, GateStatus
from risk.models import RiskCheckResult
from scoring.models import Direction, Tier, TradeRecommendation
from validators.models import TechnicalIndicators, TechnicalValidation, ValidatedSignal, ValidationStatus


def make_validated_signal(should_trade: bool = True) -> ValidatedSignal:
    """Create a test validated signal."""
    indicators = TechnicalIndicators(rsi=50.0, macd_histogram=0.5)
    validation = TechnicalValidation(
        status=ValidationStatus.PASS if should_trade else ValidationStatus.VETO,
        indicators=indicators,
        veto_reasons=[] if should_trade else ["RSI overbought"],
    )
    return ValidatedSignal(
        original=make_analyzed_message(),
        validation=validation,
    )


def make_recommendation(tier: Tier = Tier.TIER_1) -> TradeRecommendation:
    """Create a test trade recommendation."""
    return TradeRecommendation(
        symbol="AAPL",
        direction=Direction.LONG,
        tier=tier,
        score=85.0,
        confidence=0.9,
        stop_loss=145.0,
        take_profit=165.0,
        position_size=10,
        reasons=["Strong sentiment"],
        components={},
    )


def make_risk_result(approved: bool = True) -> RiskCheckResult:
    """Create a test risk check result."""
    return RiskCheckResult(
        approved=approved,
        adjusted_quantity=10,
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
        return TradingOrchestrator(**mock_components, settings=settings)

    @pytest.mark.asyncio
    async def test_process_immediate_full_flow(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Full pipeline executes successfully."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(Tier.TIER_1)
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
        mock_components["signal_scorer"].score.return_value = make_recommendation(Tier.NO_TRADE)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "low_score"
        mock_components["trade_executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_immediate_risk_rejected(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Risk rejection prevents execution."""
        mock_components["technical_validator"].validate.return_value = make_validated_signal(True)
        mock_components["signal_scorer"].score.return_value = make_recommendation(Tier.TIER_1)
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
        mock_components["signal_scorer"].score.return_value = make_recommendation(Tier.TIER_1)
        mock_components["risk_manager"].check_trade.return_value = make_risk_result(True)
        mock_components["market_gate"].check.return_value = make_gate_status(False)

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "gate_closed"
        mock_components["trade_executor"].execute.assert_not_called()
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestImmediateProcessing -v
```

**Step 3: Implement _process_immediate**

Update `_process_immediate` in `src/orchestrator/trading_orchestrator.py`:

```python
    async def _process_immediate(self, analyzed: AnalyzedMessage) -> ProcessResult:
        """Process high-signal message immediately through full pipeline.

        Args:
            analyzed: Analyzed message to process.

        Returns:
            ProcessResult with outcome.
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

        if recommendation.tier == Tier.NO_TRADE:
            return ProcessResult(status="low_score", symbol=symbol, recommendation=recommendation)

        # Step 3: Risk check
        try:
            risk_result = self._risk_manager.check_trade(
                recommendation,
                recommendation.position_size,
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
        """Extract primary symbol from analyzed message."""
        tickers = analyzed.get_tickers()
        return tickers[0] if tickers else None

    def _get_current_price(self, symbol: str | None) -> float:
        """Get current price for symbol (placeholder)."""
        # TODO: Implement actual price fetching
        return 150.0

    def _create_unvalidated_signal(self, analyzed: AnalyzedMessage) -> ValidatedSignal:
        """Create a pass-through signal when validator fails."""
        from validators.models import TechnicalIndicators, TechnicalValidation, ValidatedSignal, ValidationStatus

        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=TechnicalIndicators(rsi=50.0),
            veto_reasons=[],
            warnings=["Validation skipped due to error"],
        )
        return ValidatedSignal(original=analyzed, validation=validation)
```

Add import at top:
```python
from scoring.models import Tier
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestImmediateProcessing -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add immediate processing pipeline"
```

---

## Task 8: Batch Processing

**Files:**
- Modify: `src/orchestrator/trading_orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

**Step 1: Add batch processing tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
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
        msg1.original.raw_data["tickers"] = ["AAPL"]
        msg2 = make_analyzed_message()
        msg2.original.raw_data["tickers"] = ["TSLA"]
        msg3 = make_analyzed_message()
        msg3.original.raw_data["tickers"] = ["AAPL"]

        orchestrator._message_buffer = [msg1, msg2, msg3]
        grouped = orchestrator._group_by_symbol()

        assert "AAPL" in grouped
        assert "TSLA" in grouped
        assert len(grouped["AAPL"]) == 2
        assert len(grouped["TSLA"]) == 1
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestBatchProcessing -v
```

**Step 3: Implement batch processing methods**

Add to `src/orchestrator/trading_orchestrator.py`:

```python
    async def _process_batch(self) -> None:
        """Process current batch buffer.

        Groups messages by symbol, aggregates sentiment,
        and processes symbols with sufficient consensus.
        """
        if not self._message_buffer:
            return

        messages = self._message_buffer.copy()
        self._message_buffer.clear()

        logger.info(f"Processing batch of {len(messages)} messages")

        # Group by symbol
        by_symbol = self._group_by_symbol_from_list(messages)

        # Process each symbol
        for symbol, symbol_messages in by_symbol.items():
            aggregated = self._aggregate_sentiment(symbol, symbol_messages)

            # Only process if consensus is strong enough
            if aggregated.consensus_strength >= self._settings.min_consensus:
                await self._process_symbol_batch(symbol, aggregated)
            else:
                logger.debug(f"Skipping {symbol}: consensus {aggregated.consensus_strength:.2f} < {self._settings.min_consensus}")

    def _group_by_symbol(self) -> dict[str, list[AnalyzedMessage]]:
        """Group buffer messages by symbol."""
        return self._group_by_symbol_from_list(self._message_buffer)

    def _group_by_symbol_from_list(self, messages: list[AnalyzedMessage]) -> dict[str, list[AnalyzedMessage]]:
        """Group messages by their primary symbol."""
        grouped: dict[str, list[AnalyzedMessage]] = {}

        for msg in messages:
            tickers = msg.get_tickers()
            if tickers:
                symbol = tickers[0]
                if symbol not in grouped:
                    grouped[symbol] = []
                grouped[symbol].append(msg)

        return grouped

    def _aggregate_sentiment(self, symbol: str, messages: list[AnalyzedMessage]) -> AggregatedSentiment:
        """Aggregate sentiments from multiple messages.

        Args:
            symbol: Stock symbol.
            messages: List of analyzed messages.

        Returns:
            AggregatedSentiment with consensus metrics.
        """
        bullish = sum(1 for m in messages if m.sentiment_result.label == "bullish")
        bearish = sum(1 for m in messages if m.sentiment_result.label == "bearish")
        neutral = sum(1 for m in messages if m.sentiment_result.label == "neutral")
        total = len(messages)

        # Determine consensus
        counts = {"bullish": bullish, "bearish": bearish, "neutral": neutral}
        consensus_label = max(counts, key=counts.get)
        consensus_count = counts[consensus_label]
        consensus_strength = consensus_count / total if total > 0 else 0.0

        # Average confidence
        avg_confidence = sum(m.sentiment_result.confidence for m in messages) / total if total > 0 else 0.0

        return AggregatedSentiment(
            symbol=symbol,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            total_count=total,
            consensus_label=consensus_label,
            consensus_strength=consensus_strength,
            avg_confidence=avg_confidence,
        )

    async def _process_symbol_batch(self, symbol: str, aggregated: AggregatedSentiment) -> None:
        """Process aggregated batch for a symbol.

        Creates synthetic signal from aggregated sentiment and processes through pipeline.
        """
        logger.info(f"Processing batch for {symbol}: {aggregated.consensus_label} ({aggregated.consensus_strength:.2%})")
        # TODO: Create synthetic analyzed message from aggregation and process
        # For now, just log
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestBatchProcessing -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "feat(orchestrator): add batch processing logic"
```

---

## Task 9: Graceful Degradation

**Files:**
- Modify: `src/orchestrator/trading_orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

**Step 1: Add graceful degradation tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
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
        return TradingOrchestrator(**mock_components, settings=settings)

    @pytest.mark.asyncio
    async def test_validator_error_continues_when_enabled(
        self, orchestrator: TradingOrchestrator, mock_components: dict
    ) -> None:
        """Validator error doesn't stop processing when continue_without_validator=True."""
        mock_components["technical_validator"].validate.side_effect = Exception("Validator down")
        mock_components["signal_scorer"].score.return_value = make_recommendation(Tier.TIER_1)
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
        mock_components["signal_scorer"].score.return_value = make_recommendation(Tier.TIER_1)
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
        orchestrator = TradingOrchestrator(**mock_components, settings=settings)

        mock_components["technical_validator"].validate.side_effect = Exception("Validator down")

        result = await orchestrator._process_immediate(make_analyzed_message())

        assert result.status == "error"
        assert "Validation error" in result.error
```

**Step 2: Run tests to verify they pass**

The implementation already handles these cases. Run:

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_orchestrator.py::TestGracefulDegradation -v
```

Expected: All 3 tests PASS

**Step 3: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add src/orchestrator/ tests/orchestrator/ && git commit -m "test(orchestrator): add graceful degradation tests"
```

---

## Task 10: Integration Tests

**Files:**
- Create: `tests/orchestrator/test_integration.py`

**Step 1: Create integration tests**

```python
# tests/orchestrator/test_integration.py
"""Integration tests for trading orchestrator."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from analyzers.analyzed_message import AnalyzedMessage
from analyzers.sentiment_result import SentimentResult
from execution.models import ExecutionResult
from gate.models import GateCheckResult, GateStatus
from models.social_message import SocialMessage, SourceType
from orchestrator.models import OrchestratorState
from orchestrator.settings import OrchestratorSettings
from orchestrator.trading_orchestrator import TradingOrchestrator
from risk.models import RiskCheckResult
from scoring.models import Direction, Tier, TradeRecommendation
from validators.models import TechnicalIndicators, TechnicalValidation, ValidatedSignal, ValidationStatus


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
    def full_mock_components(self) -> dict:
        """Create fully configured mock components."""
        collector = MagicMock()
        collector.connect_all = AsyncMock()
        collector.disconnect_all = AsyncMock()
        collector.add_callback = MagicMock()

        async def mock_stream():
            yield SocialMessage(
                source=SourceType.TWITTER,
                content="AAPL to the moon!",
                author="trader1",
                timestamp=datetime.now(),
                raw_data={"tickers": ["AAPL"]},
            )

        collector.stream_all = mock_stream

        analyzer = AsyncMock()
        analyzer.analyze.return_value = AnalyzedMessage(
            original=SocialMessage(
                source=SourceType.TWITTER,
                content="AAPL to the moon!",
                author="trader1",
                timestamp=datetime.now(),
                raw_data={"tickers": ["AAPL"]},
            ),
            sentiment_result=SentimentResult(
                label="bullish",
                confidence=0.9,
                scores={"bullish": 0.9},
            ),
        )

        validator = AsyncMock()
        validator.validate.return_value = ValidatedSignal(
            original=analyzer.analyze.return_value,
            validation=TechnicalValidation(
                status=ValidationStatus.PASS,
                indicators=TechnicalIndicators(rsi=55.0),
                veto_reasons=[],
            ),
        )

        scorer = MagicMock()
        scorer.score.return_value = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            tier=Tier.TIER_1,
            score=85.0,
            confidence=0.9,
            stop_loss=145.0,
            take_profit=165.0,
            position_size=10,
            reasons=["Strong bullish sentiment"],
            components={},
        )

        risk = MagicMock()
        risk.check_trade.return_value = RiskCheckResult(
            approved=True,
            adjusted_quantity=10,
            rejection_reason=None,
        )

        gate = AsyncMock()
        gate.check.return_value = GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="all", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )

        executor = AsyncMock()
        executor.execute.return_value = ExecutionResult(
            success=True,
            order_id="ORDER-123",
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=150.0,
            error_message=None,
            timestamp=datetime.now(),
        )

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
```

**Step 2: Run integration tests**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest tests/orchestrator/test_integration.py -v
```

Expected: All tests PASS

**Step 3: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git add tests/orchestrator/ && git commit -m "test(orchestrator): add integration tests"
```

---

## Task 11: Final Test Suite Run

**Step 1: Run full test suite with coverage**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && uv run pytest --cov=src --cov-report=term-missing -v
```

Expected: All tests pass, coverage >90% on orchestrator module

**Step 2: Verify orchestrator module coverage**

Check that:
- `src/orchestrator/models.py` - 100%
- `src/orchestrator/settings.py` - 100%
- `src/orchestrator/trading_orchestrator.py` - >85%

**Step 3: Final commit if any cleanup needed**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase8-orchestrator && git status
```

If clean, proceed to merge.
