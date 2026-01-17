# Phase 8: Trading Orchestrator Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Main orchestrator that coordinates all pipeline components with hybrid processing mode.

**Architecture:** TradingOrchestrator manages collector → analyzer → validator → scorer → risk → gate → executor flow with graceful degradation.

**Tech Stack:** Python asyncio, dataclasses, Pydantic settings

---

## Design Decisions

- **Hybrid processing** - High-signal messages (confidence >= 0.85) processed immediately, others batched
- **Graceful degradation** - Continue with available components, log failures
- **Manual + Continuous** - Start/stop commands, runs continuously when active
- **Direct calls** - Simple pipeline flow, easier to debug

---

## Data Models

### OrchestratorState

```python
class OrchestratorState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"
```

### ProcessResult

```python
@dataclass
class ProcessResult:
    status: str  # "executed", "vetoed", "low_score", "gate_closed", "error", "buffered"
    symbol: str | None = None
    recommendation: TradeRecommendation | None = None
    execution_result: ExecutionResult | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## TradingOrchestrator Component

```python
class TradingOrchestrator:
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

    async def start(self) -> None:
        """Start the orchestrator - begins processing."""

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""

    async def _on_message(self, message: SocialMessage) -> ProcessResult:
        """Callback when message arrives from collector."""

    def _is_high_signal(self, msg: AnalyzedMessage) -> bool:
        """Determine if message requires immediate processing."""

    async def _process_immediate(self, analyzed: AnalyzedMessage) -> ProcessResult:
        """Full pipeline for high-signal message."""

    async def _batch_processor(self) -> None:
        """Task that processes buffer every N seconds."""

    async def _process_symbol_batch(self, symbol: str, messages: list[AnalyzedMessage]) -> None:
        """Process batch of messages for a symbol."""

    def _aggregate_sentiment(self, messages: list[AnalyzedMessage]) -> AggregatedSentiment:
        """Aggregate sentiments from multiple messages."""

    @property
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._state == OrchestratorState.RUNNING
```

---

## Processing Flow

### Hybrid Mode

```
Message arrives
    │
    ├── High-signal (confidence >= 0.85, not neutral)
    │       │
    │       └── Immediate processing → Full pipeline
    │
    └── Regular signal
            │
            └── Add to buffer → Batch process every 60s
```

### Immediate Processing Pipeline

```
_process_immediate(analyzed)
    │
    ├── 1. Validate technically
    │       └── If vetoed → return "vetoed"
    │
    ├── 2. Score signal
    │       └── If NO_TRADE tier → return "low_score"
    │
    ├── 3. Check risk
    │       └── If not approved → return "risk_rejected"
    │
    ├── 4. Check gate
    │       └── If closed → return "gate_closed"
    │
    └── 5. Execute
            └── return "executed" with result
```

### Batch Processing

```
Every 60 seconds:
    │
    ├── Take messages from buffer
    │
    ├── Group by symbol
    │
    └── For each symbol:
            │
            ├── Aggregate sentiments
            │
            ├── Check consensus strength
            │       └── If < min_consensus → skip
            │
            └── Process through pipeline
```

---

## Graceful Degradation

| Component | Critical | Fallback on Error |
|-----------|----------|-------------------|
| Collector | Yes | Stop orchestrator |
| Analyzer | Yes | Skip message |
| Validator | No | Unvalidated signal (pass through) |
| Scorer | Yes | Skip message |
| Risk | Yes | Skip trade |
| Gate | No | Closed (fail-safe) |
| Executor | Yes | Return error result |

---

## Settings

```python
class OrchestratorSettings(BaseModel):
    enabled: bool = True

    # Hybrid mode thresholds
    immediate_threshold: float = 0.85  # Confidence for immediate processing

    # Batch settings
    batch_interval_seconds: int = 60   # How often to process batch
    min_consensus: float = 0.6         # Minimum consensus to act on batch
    max_buffer_size: int = 1000        # Buffer limit

    # Graceful degradation
    continue_without_validator: bool = True
    gate_fail_safe_closed: bool = True
```

Added to `config/settings.yaml`:

```yaml
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

---

## File Structure

```
src/orchestrator/
├── __init__.py              # Exports
├── models.py                # OrchestratorState, ProcessResult, AggregatedSentiment
├── settings.py              # OrchestratorSettings
└── trading_orchestrator.py  # TradingOrchestrator class

tests/orchestrator/
├── __init__.py
├── test_models.py           # Model tests
├── test_orchestrator.py     # Unit tests
└── test_integration.py      # Integration tests
```

---

## Test Coverage

### Unit Tests (test_models.py)
- OrchestratorState enum values
- ProcessResult creation with different statuses
- AggregatedSentiment calculation

### Unit Tests (test_orchestrator.py)
- start() changes state to RUNNING
- stop() changes state to STOPPED
- _is_high_signal returns True for high confidence
- _is_high_signal returns False for neutral
- _is_high_signal returns False for low confidence
- _on_message routes high-signal to immediate
- _on_message routes regular to buffer
- _process_immediate calls all pipeline stages
- _process_immediate handles validator error gracefully
- _process_immediate handles gate error with fail-safe
- _batch_processor processes buffer periodically
- _aggregate_sentiment calculates consensus
- Buffer respects max_buffer_size

### Integration Tests (test_integration.py)
- Full flow: message → execution
- Batch processing with multiple symbols
- Graceful degradation with failing validator
- Gate closed blocks execution
