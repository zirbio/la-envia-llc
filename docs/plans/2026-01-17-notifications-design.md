# Phase 9: Notifications (Telegram) Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Telegram notifications for trading alerts and pre-market checklist.

**Architecture:** TelegramNotifier sends formatted alerts via python-telegram-bot. AlertFormatter converts pipeline data to readable messages. ChecklistHandler manages interactive pre-market checklist.

**Tech Stack:** Python asyncio, python-telegram-bot, Pydantic settings

---

## Design Decisions

- **Async bot** - Non-blocking message sending
- **Separated formatters** - Clean separation between data and presentation
- **Interactive checklist** - Inline keyboard buttons for item confirmation
- **Graceful degradation** - Log errors but don't crash pipeline if Telegram fails

---

## Data Models

### AlertType

```python
class AlertType(Enum):
    NEW_SIGNAL = "new_signal"
    ENTRY_EXECUTED = "entry_executed"
    EXIT_EXECUTED = "exit_executed"
    CIRCUIT_BREAKER = "circuit_breaker"
    DAILY_SUMMARY = "daily_summary"
```

### Alert

```python
@dataclass
class Alert:
    alert_type: AlertType
    symbol: str | None = None
    message: str = ""
    data: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## Components

### AlertFormatter

Formats pipeline data into readable Telegram messages with markdown.

```python
class AlertFormatter:
    def format_new_signal(self, recommendation: TradeRecommendation) -> str:
        """Format a new trading signal alert."""

    def format_execution(self, result: ExecutionResult, is_entry: bool) -> str:
        """Format an execution result (entry or exit)."""

    def format_circuit_breaker(self, reason: str, risk_state: DailyRiskState) -> str:
        """Format a circuit breaker trigger alert."""

    def format_daily_summary(self, stats: dict) -> str:
        """Format daily trading summary."""
```

### TelegramNotifier

Main notification class that sends alerts via Telegram.

```python
class TelegramNotifier:
    def __init__(
        self,
        settings: NotificationSettings,
        formatter: AlertFormatter,
    ):
        self._settings = settings
        self._formatter = formatter
        self._bot: Bot | None = None

    async def start(self) -> None:
        """Initialize the Telegram bot."""

    async def stop(self) -> None:
        """Shutdown the bot gracefully."""

    async def send_alert(self, alert: Alert) -> bool:
        """Send a generic alert. Returns True if sent successfully."""

    async def send_new_signal(self, recommendation: TradeRecommendation) -> bool:
        """Send a new signal alert."""

    async def send_execution(self, result: ExecutionResult, is_entry: bool) -> bool:
        """Send an execution alert (entry or exit)."""

    async def send_circuit_breaker(self, reason: str, risk_state: DailyRiskState) -> bool:
        """Send a circuit breaker alert."""

    async def send_daily_summary(self, stats: dict) -> bool:
        """Send daily summary."""

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled and configured."""
```

### ChecklistHandler

Handles interactive pre-market checklist with inline keyboard buttons.

```python
class ChecklistHandler:
    def __init__(
        self,
        settings: NotificationSettings,
        bot: Bot,
        chat_id: str,
    ):
        self._settings = settings
        self._bot = bot
        self._chat_id = chat_id
        self._checked_items: set[int] = set()
        self._message_id: int | None = None

    async def send_checklist(self) -> None:
        """Send the pre-market checklist with inline buttons."""

    async def on_item_checked(self, item_index: int) -> None:
        """Handle when user checks an item. Updates message with checked state."""

    def is_checklist_complete(self) -> bool:
        """Check if all items have been checked."""

    def reset_checklist(self) -> None:
        """Reset checklist for next day."""

    def _build_keyboard(self) -> InlineKeyboardMarkup:
        """Build inline keyboard with current checked states."""

    def _format_checklist_message(self) -> str:
        """Format the checklist message showing checked/unchecked items."""
```

---

## Message Formats

### New Signal

```
🚨 NEW SIGNAL: AAPL

📊 Direction: LONG
⭐ Score: 85/100 (STRONG)
💰 Entry: $178.50
🛑 Stop: $176.00 (-1.4%)
🎯 Target: $183.00 (+2.5%)
📈 R:R: 1:1.8

📝 High sentiment confidence from multiple sources
```

### Entry Executed

```
✅ ENTRY EXECUTED

📈 AAPL - LONG
📦 Quantity: 50 shares
💵 Price: $178.52
💰 Value: $8,926.00
```

### Exit Executed

```
🏁 EXIT EXECUTED

📉 AAPL - SOLD
📦 Quantity: 50 shares
💵 Price: $181.20
💰 P&L: +$134.00 (+1.5%)
```

### Circuit Breaker Triggered

```
🚫 CIRCUIT BREAKER TRIGGERED

⚠️ Reason: Daily loss limit reached
📉 Daily P&L: -$450.00 (-3.0%)
📊 Trades today: 5
🔒 Trading blocked until tomorrow
```

### Daily Summary

```
📊 DAILY SUMMARY - 2026-01-17

📈 Trades: 5
✅ Winners: 3 (60%)
❌ Losers: 2

💰 Gross P&L: +$285.00
📉 Largest Win: +$180.00 (AAPL)
📉 Largest Loss: -$95.00 (MSFT)

⭐ Profit Factor: 1.8
```

### Pre-Market Checklist

```
📋 PRE-MARKET CHECKLIST

Ready for trading day? Complete all items:

☐ Economic calendar reviewed
☐ Overnight news checked
☐ Watchlist prepared
☐ Mental state: focused
☐ Risk parameters confirmed

[Button: Item 1] [Button: Item 2] ...
```

After completion:
```
✅ CHECKLIST COMPLETE

All items verified. Ready to trade!
```

---

## Settings

```python
class NotificationSettings(BaseModel):
    enabled: bool = True
    telegram_token: str = ""  # From TELEGRAM_BOT_TOKEN env
    chat_id: str = ""  # From TELEGRAM_CHAT_ID env

    # Which alert types to send
    alert_types: list[str] = [
        "new_signal",
        "entry_executed",
        "exit_executed",
        "circuit_breaker",
        "daily_summary"
    ]

    # Pre-market checklist
    pre_market_checklist_enabled: bool = True
    pre_market_checklist_time: str = "09:00"
    checklist_items: list[str] = [
        "Economic calendar reviewed",
        "Overnight news checked",
        "Watchlist prepared",
        "Mental state: focused",
        "Risk parameters confirmed"
    ]
```

Added to `config/settings.yaml`:

```yaml
# Phase 9: Notifications
notifications:
  enabled: true
  # telegram_token and chat_id from environment variables
  alert_types:
    - new_signal
    - entry_executed
    - exit_executed
    - circuit_breaker
    - daily_summary
  pre_market_checklist_enabled: true
  pre_market_checklist_time: "09:00"
  checklist_items:
    - "Economic calendar reviewed"
    - "Overnight news checked"
    - "Watchlist prepared"
    - "Mental state: focused"
    - "Risk parameters confirmed"
```

---

## File Structure

```
src/notifications/
├── __init__.py              # Exports
├── models.py                # AlertType, Alert
├── settings.py              # NotificationSettings
├── alert_formatter.py       # AlertFormatter class
├── telegram_notifier.py     # TelegramNotifier class
└── checklist_handler.py     # ChecklistHandler class

tests/notifications/
├── __init__.py
├── test_models.py           # Model tests
├── test_formatter.py        # Formatter tests
├── test_notifier.py         # Notifier tests (mocked bot)
└── test_checklist.py        # Checklist tests (mocked bot)
```

---

## Integration with Orchestrator

The `TradingOrchestrator` will be extended to call the notifier:

```python
# In TradingOrchestrator._process_immediate():
if result.status == "executed" and result.execution_result:
    await self._notifier.send_execution(result.execution_result, is_entry=True)

# In TradingOrchestrator (when recommendation generated):
if recommendation.tier in (ScoreTier.STRONG, ScoreTier.MODERATE):
    await self._notifier.send_new_signal(recommendation)
```

---

## Test Coverage

### Unit Tests (test_models.py)
- AlertType enum values
- Alert creation with defaults
- Alert creation with all fields

### Unit Tests (test_formatter.py)
- format_new_signal with LONG direction
- format_new_signal with SHORT direction
- format_execution for entry
- format_execution for exit
- format_circuit_breaker
- format_daily_summary with wins and losses
- format_daily_summary with no trades

### Unit Tests (test_notifier.py)
- start() initializes bot
- stop() shuts down gracefully
- send_alert returns True on success
- send_alert returns False on error (graceful degradation)
- send_new_signal formats and sends
- send_execution for entry
- send_execution for exit
- send_circuit_breaker
- send_daily_summary
- is_enabled returns False when token missing

### Unit Tests (test_checklist.py)
- send_checklist sends message with keyboard
- on_item_checked updates message
- is_checklist_complete returns False when items remain
- is_checklist_complete returns True when all checked
- reset_checklist clears state
- _build_keyboard shows checked/unchecked states
