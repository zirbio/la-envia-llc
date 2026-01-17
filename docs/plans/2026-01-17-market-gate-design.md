# Phase 7: Market Condition Gate Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Gate that verifies market conditions before allowing trading. Soft block = process signals but don't execute.

**Architecture:** MarketGate evaluates 4 independent conditions, returns GateStatus with position_size_factor.

**Tech Stack:** Python dataclasses, Alpaca API (volume), yfinance (VIX), Pydantic settings

---

## Design Decisions

- **Soft block with logging** - Process signals but don't execute trades when gate closed (useful for backtesting/analysis)
- **VIX elevated handling** - Reduce position size by 0.5x when VIX between 25-30
- **Data sources** - Alpaca for SPY/QQQ volume, yfinance for VIX
- **Choppy market detection** - Simple ATR ratio: (High-Low) / ATR < 1.5

---

## Data Models

### GateCheckResult

Result of a single gate check.

```python
@dataclass
class GateCheckResult:
    name: str                    # "trading_hours", "volume", "vix", "choppy"
    passed: bool                 # Did the check pass?
    reason: str | None           # Reason if failed
    data: dict                   # Observed data (vix_value, volume, etc.)
```

### GateStatus

Combined result of all gate checks.

```python
@dataclass
class GateStatus:
    timestamp: datetime
    is_open: bool                # Did all checks pass?
    checks: list[GateCheckResult]
    position_size_factor: float  # 1.0 normal, 0.5 if VIX elevated, 0.0 if blocked

    def get_failed_checks(self) -> list[GateCheckResult]:
        """Return checks that did not pass."""
        return [c for c in self.checks if not c.passed]
```

---

## MarketGate Component

Main class that evaluates market conditions.

```python
class MarketGate:
    def __init__(
        self,
        alpaca_client: AlpacaClient,
        settings: MarketGateSettings,
    ):
        self._alpaca = alpaca_client
        self._settings = settings

    async def check(self) -> GateStatus:
        """Evaluate all market conditions.

        Runs 4 checks in parallel and combines results.
        """

    async def _check_trading_hours(self) -> GateCheckResult:
        """Verify market hours (09:30-16:00 ET).

        Optionally avoid lunch hours (11:30-14:00).
        """

    async def _check_volume(self) -> GateCheckResult:
        """Verify minimum volume in SPY/QQQ.

        Uses Alpaca bars to get last minute volume.
        SPY >= 500k, QQQ >= 300k.
        """

    async def _check_vix(self) -> GateCheckResult:
        """Check VIX level via yfinance.

        VIX > 30: blocked
        VIX 25-30: elevated (reduce size)
        VIX < 25: normal
        """

    async def _check_choppy_market(self) -> GateCheckResult:
        """Detect directionless market.

        Calculate (High - Low of day) / ATR(14) for SPY.
        If ratio < 1.5: choppy market.
        """

    def _calculate_position_factor(self, checks: list[GateCheckResult]) -> float:
        """Calculate size factor based on checks.

        Any check failed → 0.0
        VIX elevated → 0.5
        All normal → 1.0
        """
```

---

## Gate Flow

```
MarketGate.check() → GateStatus
    │
    ├── is_open=True, factor=1.0  → Normal trading
    ├── is_open=True, factor=0.5  → Trading with reduced size (VIX elevated)
    └── is_open=False, factor=0.0 → Soft block (process, don't execute)
```

---

## Settings

```python
class MarketGateSettings(BaseModel):
    enabled: bool = True

    # Trading hours (ET timezone)
    trading_start: str = "09:30"
    trading_end: str = "16:00"
    avoid_lunch: bool = True
    lunch_start: str = "11:30"
    lunch_end: str = "14:00"

    # Volume requirements (1-minute bars)
    spy_min_volume: int = 500_000
    qqq_min_volume: int = 300_000

    # VIX thresholds
    vix_max: float = 30.0
    vix_elevated: float = 25.0
    elevated_size_factor: float = 0.5

    # Choppy market detection
    choppy_detection_enabled: bool = True
    choppy_atr_ratio_threshold: float = 1.5
```

Added to `config/settings.yaml`:

```yaml
# CAPA 0: Market Condition Gate
market_gate:
  enabled: true
  trading_start: "09:30"
  trading_end: "16:00"
  avoid_lunch: true
  lunch_start: "11:30"
  lunch_end: "14:00"
  spy_min_volume: 500000
  qqq_min_volume: 300000
  vix_max: 30.0
  vix_elevated: 25.0
  elevated_size_factor: 0.5
  choppy_detection_enabled: true
  choppy_atr_ratio_threshold: 1.5
```

---

## Integration with Pipeline

**Location in flow:**

```
MarketGate.check() → GateStatus
    │
    ├── is_open=False → Log signals, DON'T execute
    │
    └── is_open=True
            ↓
        Phase 1-5 (Collectors → Risk)
            ↓
        TradeExecutor.execute(
            recommendation,
            risk_result,
            gate_status  ← NEW parameter
        )
            │
            ├── Adjust quantity by position_size_factor
            └── Execute order
```

**Change in TradeExecutor:**

```python
async def execute(
    self,
    recommendation: TradeRecommendation,
    risk_result: RiskCheckResult,
    gate_status: GateStatus | None = None,  # NEW
) -> ExecutionResult:
    # Check 1: Risk approved
    if not risk_result.approved:
        return ExecutionResult(success=False, error_message="Not approved")

    # Check 2: Gate open (if provided)
    if gate_status and not gate_status.is_open:
        return ExecutionResult(
            success=False,
            error_message=f"Market gate closed: {gate_status.get_failed_checks()}"
        )

    # Adjust quantity by factor
    adjusted_quantity = risk_result.adjusted_quantity
    if gate_status:
        adjusted_quantity = int(adjusted_quantity * gate_status.position_size_factor)

    # Continue with execution...
```

---

## File Structure

```
src/gate/
├── __init__.py              # Exports
├── models.py                # GateCheckResult, GateStatus
├── market_gate.py           # MarketGate class
└── vix_fetcher.py           # yfinance wrapper for VIX

tests/gate/
├── __init__.py
├── test_models.py           # GateCheckResult, GateStatus tests
├── test_market_gate.py      # MarketGate unit tests
└── test_vix_fetcher.py      # VIX fetcher tests

tests/integration/
└── test_gate_execution.py   # Gate → Execution integration
```

---

## Test Coverage

### Unit Tests (test_models.py)
- GateCheckResult creation with passed=True
- GateCheckResult creation with passed=False
- GateStatus creation with all checks passed
- GateStatus.get_failed_checks returns failed only
- GateStatus with mixed results

### Unit Tests (test_market_gate.py)
- Trading hours pass during market hours
- Trading hours fail before 09:30
- Trading hours fail after 16:00
- Trading hours fail during lunch (if enabled)
- Trading hours pass during lunch (if disabled)
- Volume pass when SPY/QQQ above minimum
- Volume fail when SPY below minimum
- Volume fail when QQQ below minimum
- VIX normal (< 25) returns factor 1.0
- VIX elevated (25-30) returns factor 0.5
- VIX blocked (> 30) returns is_open=False
- Choppy market detected when ratio < 1.5
- Choppy market pass when ratio >= 1.5
- Full gate check with all passing
- Full gate check with one failing

### Unit Tests (test_vix_fetcher.py)
- Fetch VIX returns current value
- Fetch VIX handles network error gracefully

### Integration Tests (test_gate_execution.py)
- Executor respects gate closed (doesn't execute)
- Executor applies size factor when VIX elevated
- Full flow: Gate → Risk → Execution with factor
