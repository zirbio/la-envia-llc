# Phase 6: Position-Aware Trade Executor Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute approved trades via Alpaca bracket orders, track positions, and feed P&L back to RiskManager.

**Architecture:** TradeExecutor sits after RiskManager. It receives approved RiskCheckResult + TradeRecommendation, submits bracket orders to Alpaca, tracks open positions internally, and updates RiskManager with P&L after each trade.

**Tech Stack:** Python dataclasses, Alpaca API (alpaca-py), Pydantic settings

---

## Design Decisions

- **Market orders only** - Simplest execution, acceptable slippage for liquid intraday stocks
- **Alpaca bracket orders** - Entry + stop-loss + take-profit as single order. Alpaca handles exits automatically
- **Update P&L after each trade** - No background polling. Update unrealized P&L after each execution
- **Compare positions after trade** - Detect closed positions by comparing tracked state with Alpaca
- **Return result object** - ExecutionResult captures success/failure explicitly without exceptions

---

## Data Models

### ExecutionResult

Result of an execution attempt.

```python
@dataclass
class ExecutionResult:
    success: bool                    # Did the order submit successfully?
    order_id: str | None             # Alpaca order ID (if success)
    symbol: str                      # Stock symbol
    side: str                        # "buy" or "sell"
    quantity: int                    # Shares executed
    filled_price: float | None       # Fill price (if filled)
    error_message: str | None        # Error details (if failed)
    timestamp: datetime              # When execution was attempted
```

### TrackedPosition

Local view of an open position.

```python
@dataclass
class TrackedPosition:
    symbol: str                      # Stock symbol
    quantity: int                    # Shares held
    entry_price: float               # Average entry price
    entry_time: datetime             # When position was opened
    stop_loss: float                 # Stop loss price
    take_profit: float               # Take profit price
    order_id: str                    # Original order ID
    direction: Direction             # LONG or SHORT
```

---

## TradeExecutor Component

Main class that executes trades and tracks positions.

```python
class TradeExecutor:
    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_manager: RiskManager,
    ):
        self._alpaca = alpaca_client
        self._risk_manager = risk_manager
        self._tracked_positions: dict[str, TrackedPosition] = {}

    async def execute(
        self,
        recommendation: TradeRecommendation,
        risk_result: RiskCheckResult,
    ) -> ExecutionResult:
        """Execute an approved trade as a bracket order.

        Steps:
        1. Validate risk_result is approved
        2. Detect any closed positions (compare with Alpaca)
        3. Submit bracket order (entry + stop-loss + take-profit)
        4. Track the new position locally
        5. Update RiskManager with unrealized P&L
        6. Return ExecutionResult
        """

    async def sync_positions(self) -> list[str]:
        """Compare tracked positions with Alpaca, detect closes.

        Returns list of symbols that were closed.
        For each closed position, calculates realized P&L
        and calls risk_manager.record_close().
        """

    async def get_unrealized_pnl(self) -> float:
        """Fetch current unrealized P&L from Alpaca positions."""

    def get_tracked_positions(self) -> dict[str, TrackedPosition]:
        """Return current locally tracked positions."""
```

---

## Execution Flow

```
execute(recommendation, risk_result)
    │
    ├── 1. Check risk_result.approved == True
    │       └── If False: return ExecutionResult(success=False, error="Not approved")
    │
    ├── 2. sync_positions()
    │       └── Detect closed positions, record realized P&L
    │
    ├── 3. Submit bracket order via Alpaca
    │       └── If error: return ExecutionResult(success=False, error=message)
    │
    ├── 4. Track position locally in _tracked_positions
    │
    ├── 5. Update unrealized P&L to RiskManager
    │
    └── 6. Return ExecutionResult(success=True, order_id=..., ...)
```

---

## Position Sync Logic

```python
async def sync_positions(self) -> list[str]:
    """Detect positions closed by Alpaca (stop-loss or take-profit hit)."""

    # 1. Fetch current positions from Alpaca
    alpaca_positions = await self._alpaca.get_all_positions()
    alpaca_symbols = {p["symbol"] for p in alpaca_positions}

    # 2. Find tracked positions no longer in Alpaca
    closed_symbols = []
    for symbol, tracked in list(self._tracked_positions.items()):
        if symbol not in alpaca_symbols:
            # Position was closed (stop-loss or take-profit hit)
            closed_symbols.append(symbol)

            # 3. Calculate realized P&L
            pnl = await self._calculate_closed_pnl(tracked)

            # 4. Record to RiskManager
            self._risk_manager.record_close(symbol, pnl)

            # 5. Remove from tracking
            del self._tracked_positions[symbol]

    return closed_symbols
```

**P&L Calculation:**

```python
async def _calculate_closed_pnl(self, position: TrackedPosition) -> float:
    """Calculate realized P&L for a closed position.

    For LONG: pnl = (exit_price - entry_price) * quantity
    For SHORT: pnl = (entry_price - exit_price) * quantity

    Exit price fetched from Alpaca's filled order history.
    """
```

---

## Integration Flow

```
Phase 2 (Sentiment) → AnalyzedMessage
    ↓
Phase 3 (Technical) → ValidatedSignal
    ↓
Phase 4 (Scoring) → TradeRecommendation
    ↓
Phase 5 (Risk) → RiskCheckResult
    ↓
Phase 6 (Execution) → ExecutionResult   ← NEW
    ↓
    └── Updates RiskManager with P&L (feedback loop)
```

---

## Settings

```python
class ExecutionSettings(BaseModel):
    enabled: bool = True
    paper_mode: bool = True              # Paper vs live trading
    default_time_in_force: str = "day"   # day, gtc, ioc, fok
```

Added to `config/settings.yaml`:

```yaml
# CAPA 6: Execution
execution:
  enabled: true
  paper_mode: true
  default_time_in_force: "day"
```

---

## File Structure

```
src/execution/
├── __init__.py              # Exports
├── alpaca_client.py         # Already exists
├── models.py                # ExecutionResult, TrackedPosition
└── trade_executor.py        # TradeExecutor class

tests/execution/
├── __init__.py
├── test_alpaca_client.py    # Already exists
├── test_models.py           # ExecutionResult, TrackedPosition tests
└── test_trade_executor.py   # TradeExecutor unit tests

tests/integration/
└── test_execution_pipeline.py  # End-to-end with risk → execution
```

---

## Test Coverage

### Unit Tests (test_trade_executor.py)
- Execute approved trade returns success result
- Execute rejected trade returns failure (not approved)
- Bracket order submitted with correct stop-loss/take-profit
- Position tracked after successful execution
- Sync detects closed position and records P&L
- Sync handles multiple closed positions
- Unrealized P&L updated after execution
- Error handling when Alpaca rejects order
- LONG vs SHORT direction handled correctly

### Unit Tests (test_models.py)
- ExecutionResult creation with success
- ExecutionResult creation with failure
- TrackedPosition creation
- TrackedPosition with LONG direction
- TrackedPosition with SHORT direction

### Integration Tests (test_execution_pipeline.py)
- Full flow: TradeRecommendation → RiskManager → TradeExecutor
- Position closes, P&L recorded, RiskManager state updated
- Daily loss limit blocks subsequent trades after losses
- Multiple trades accumulate correctly
