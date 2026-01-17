# Phase 5: Simple Risk Management Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add portfolio-level risk controls that gate trade execution based on position size and daily loss limits.

**Architecture:** Simple gatekeeper pattern - RiskManager validates each TradeRecommendation before execution, returning approved/blocked status with optional warnings.

**Tech Stack:** Python dataclasses, Pydantic settings

---

## Overview

Phase 5 sits between scoring (Phase 4) and execution. It enforces two simple rules:

1. **Max position size** - No trade exceeds configured dollar amount
2. **Daily loss limit** - Stop trading after configured realized losses

**Design Decisions:**
- Dollar-based limits (not percentage)
- Hard block when limits hit (not soft warning)
- Realized losses trigger blocks; unrealized tracked separately for warnings

---

## Data Models

### RiskCheckResult

Result of validating a trade against risk rules.

```python
@dataclass
class RiskCheckResult:
    approved: bool              # Can this trade proceed?
    adjusted_quantity: int      # Final quantity (0 if blocked)
    adjusted_value: float       # Final dollar value
    rejection_reason: str | None  # Why rejected (if blocked)
    warnings: list[str]         # Non-blocking warnings (e.g., unrealized drawdown)
```

### DailyRiskState

Tracks running totals for the trading day.

```python
@dataclass
class DailyRiskState:
    date: date                  # Trading date
    realized_pnl: float         # Sum of closed position P&L
    unrealized_pnl: float       # Current open position P&L
    trades_today: int           # Number of trades executed
    is_blocked: bool            # True if daily limit hit
```

---

## RiskManager Component

Main class that validates trades against risk rules.

```python
class RiskManager:
    def __init__(
        self,
        max_position_value: float,    # e.g., 1000.0 ($1,000 max per trade)
        max_daily_loss: float,        # e.g., 500.0 ($500 max daily loss)
        unrealized_warning_threshold: float = 300.0,  # Warn at this unrealized loss
    ):
        self._max_position_value = max_position_value
        self._max_daily_loss = max_daily_loss
        self._unrealized_warning_threshold = unrealized_warning_threshold
        self._daily_state: DailyRiskState

    def check_trade(
        self,
        recommendation: TradeRecommendation,
        requested_quantity: int,
    ) -> RiskCheckResult:
        """Validate a trade against risk rules.

        Checks:
        1. Is daily limit already hit? → Block
        2. Does position value exceed max? → Block
        3. Is unrealized drawdown high? → Warn (don't block)
        """

    def record_trade(self, symbol: str, quantity: int, price: float) -> None:
        """Record an executed trade."""

    def record_close(self, symbol: str, pnl: float) -> None:
        """Record a closed position and its realized P&L.

        If realized_pnl drops below -max_daily_loss, sets is_blocked=True.
        """

    def update_unrealized_pnl(self, total_unrealized: float) -> None:
        """Update current unrealized P&L (called periodically)."""

    def reset_daily_state(self) -> None:
        """Reset for new trading day (call at market open)."""

    def get_daily_state(self) -> DailyRiskState:
        """Get current daily risk state for monitoring."""
```

---

## Integration Flow

```
Phase 2 (Sentiment)
    ↓
Phase 3 (Technical Validation)
    ↓
Phase 4 (Scoring) → TradeRecommendation
    ↓
Phase 5 (Risk) → RiskCheckResult   ← NEW
    ↓
Execution (if approved)
```

### Usage Example

```python
# After Phase 4 produces a recommendation
recommendation = signal_scorer.score(validated_signal, current_price)

# Phase 5 validates against risk rules
requested_qty = calculate_shares(
    recommendation.position_size_percent,
    portfolio_value,
    current_price
)
risk_result = risk_manager.check_trade(recommendation, requested_qty)

if risk_result.approved:
    # Execute the trade with adjusted quantity
    execute_trade(recommendation.symbol, risk_result.adjusted_quantity)
    risk_manager.record_trade(
        recommendation.symbol,
        risk_result.adjusted_quantity,
        current_price
    )
else:
    # Log rejection reason
    log_rejected(recommendation.symbol, risk_result.rejection_reason)

# Show any warnings (e.g., high unrealized drawdown)
for warning in risk_result.warnings:
    log_warning(warning)
```

---

## Settings

```python
class RiskSettings(BaseModel):
    enabled: bool = True
    max_position_value: float = Field(default=1000.0, gt=0)
    max_daily_loss: float = Field(default=500.0, gt=0)
    unrealized_warning_threshold: float = Field(default=300.0, gt=0)
```

Added to `config/settings.yaml`:

```yaml
risk:
  enabled: true
  max_position_value: 1000.0
  max_daily_loss: 500.0
  unrealized_warning_threshold: 300.0
```

---

## File Structure

```
src/risk/
├── __init__.py
├── models.py              # RiskCheckResult, DailyRiskState
└── risk_manager.py        # RiskManager class

tests/risk/
├── __init__.py
├── test_models.py
└── test_risk_manager.py

tests/integration/
└── test_risk_pipeline.py  # End-to-end with scoring → risk
```

---

## Test Coverage

### Unit Tests (test_risk_manager.py)
- Trade blocked when daily loss limit hit
- Trade blocked when position value exceeds max
- Trade approved when within all limits
- Unrealized drawdown triggers warning (not block)
- Multiple trades accumulate correctly
- Daily state resets correctly
- Edge cases: exactly at limit, zero values

### Integration Tests (test_risk_pipeline.py)
- TradeRecommendation from Phase 4 → RiskCheckResult
- Full pipeline: sentiment → technical → scoring → risk
- Realistic trading day simulation with multiple trades
