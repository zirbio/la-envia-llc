# Journal Module Design

**Fecha:** 2026-01-17
**Estado:** Diseño Validado
**Fase:** 10

---

## 1. Overview

Auto-log all trades with full context, calculate performance metrics, and analyze trading patterns for continuous improvement.

---

## 2. Data Models

### JournalEntry

```python
@dataclass
class JournalEntry:
    trade_id: str                    # "2026-01-17-NVDA-001"
    symbol: str
    direction: Direction

    # Entry details
    entry_time: datetime
    entry_price: float
    entry_quantity: int
    entry_reason: str                # From TradeRecommendation.reasoning
    entry_score: float               # Original signal score

    # Exit details
    exit_time: datetime | None
    exit_price: float | None
    exit_quantity: int
    exit_reason: str | None          # "stop_loss", "take_profit_1", "trailing_stop", etc.

    # Results (calculated on exit)
    pnl_dollars: float
    pnl_percent: float
    r_multiple: float                # PnL relative to initial risk

    # Context
    market_conditions: str
    score_components: ScoreComponents

    # Optional manual tags
    emotion_tag: str | None          # "confident", "fomo", "revenge", etc.
    notes: str | None
```

### TradingMetrics

```python
@dataclass
class TradingMetrics:
    period_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float              # 0-1 (e.g., 0.55 = 55%)
    profit_factor: float         # gross_profit / gross_loss
    expectancy: float            # avg PnL per trade in R

    avg_win_dollars: float
    avg_loss_dollars: float
    avg_win_r: float             # Average R-multiple for winners
    avg_loss_r: float            # Average R-multiple for losers

    total_pnl_dollars: float
    total_pnl_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float          # Annualized

    best_trade: JournalEntry | None
    worst_trade: JournalEntry | None
```

### PatternAnalysis

```python
@dataclass
class PatternAnalysis:
    # Time patterns
    best_hour: int               # 0-23, hour with highest avg PnL
    worst_hour: int
    best_day_of_week: int        # 0=Monday, 4=Friday

    # Symbol patterns
    best_symbols: list[tuple[str, float]]   # [(symbol, avg_pnl), ...]
    worst_symbols: list[tuple[str, float]]

    # Setup patterns (by entry_reason keywords)
    best_setups: list[tuple[str, float]]
    worst_setups: list[tuple[str, float]]

    # Duration patterns
    avg_winner_duration_minutes: float
    avg_loser_duration_minutes: float
```

---

## 3. Components

### TradeLogger (`trade_logger.py`)

- `log_entry(position: TrackedPosition, recommendation: TradeRecommendation)` - Called when trade opens
- `log_exit(trade_id: str, exit_price: float, exit_reason: str)` - Called when trade closes
- `add_emotion_tag(trade_id: str, tag: str)` - Manual tag from Telegram
- `add_notes(trade_id: str, notes: str)` - Manual notes
- Generates unique trade IDs: `{date}-{symbol}-{sequence}`
- Writes to `data/trades/{date}.json`

### MetricsCalculator (`metrics_calculator.py`)

- `calculate(entries: list[JournalEntry], period_days: int = 30)` → `TradingMetrics`
- Calculates: win_rate, profit_factor, expectancy, avg_win, avg_loss, max_drawdown, sharpe_ratio
- Supports filtering by symbol, direction, time period

### PatternAnalyzer (`pattern_analyzer.py`)

- `analyze(entries: list[JournalEntry])` → `PatternAnalysis`
- Identifies: best/worst trading hours, best/worst symbols, best/worst setups
- Calculates average hold time for winners vs losers

### JournalManager (`journal_manager.py`)

- Orchestrates the three components
- `on_trade_opened(position, recommendation)` - Hook from execution
- `on_trade_closed(trade_id, exit_price, reason)` - Hook from execution
- `get_daily_summary()` → For Telegram daily report
- `get_weekly_report()` → For weekly analysis

---

## 4. Settings

```python
class JournalSettings(BaseModel):
    enabled: bool = True
    data_dir: str = "data/trades"

    # Auto-logging
    auto_log_entries: bool = True
    auto_log_exits: bool = True

    # Metrics
    default_period_days: int = 30

    # Reports
    weekly_report_enabled: bool = True
    weekly_report_day: str = "saturday"
```

---

## 5. Integration Points

1. **With Execution** - TradeExecutor calls:
   - `journal_manager.on_trade_opened()` after successful entry
   - `journal_manager.on_trade_closed()` after exit fills

2. **With Notifications** - TelegramNotifier uses:
   - `journal_manager.get_daily_summary()` for daily report
   - `journal_manager.get_weekly_report()` for weekly analysis

3. **With Config** - Add to Settings class:
   - `journal: JournalSettings`

---

## 6. File Storage

```
data/
└── trades/
    ├── 2026-01-17.json
    ├── 2026-01-18.json
    └── ...
```

Each JSON file contains a list of JournalEntry objects for that day.

---

## 7. Testing Strategy

### Unit Tests

**TradeLogger:**
- `test_log_entry_creates_file`
- `test_log_entry_appends`
- `test_log_exit_updates_entry`
- `test_trade_id_generation`

**MetricsCalculator:**
- `test_win_rate_calculation`
- `test_profit_factor`
- `test_expectancy`
- `test_empty_entries`
- `test_max_drawdown`

**PatternAnalyzer:**
- `test_best_hour_detection`
- `test_best_symbol_ranking`
- `test_duration_calculation`

### Integration Tests
- `test_full_trade_lifecycle`
- `test_daily_summary_format`

---

*Documento generado: 2026-01-17*
