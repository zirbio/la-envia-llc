# Phase 9: Notifications Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Telegram notifications for trading alerts and interactive pre-market checklist.

**Architecture:** TelegramNotifier sends formatted alerts via python-telegram-bot async API. AlertFormatter converts pipeline models to readable messages. ChecklistHandler manages interactive inline keyboard buttons.

**Tech Stack:** Python asyncio, python-telegram-bot, Pydantic settings, dataclasses

---

## Task 1: Data Models (AlertType, Alert)

**Files:**
- Create: `src/notifications/__init__.py`
- Create: `src/notifications/models.py`
- Create: `tests/notifications/__init__.py`
- Create: `tests/notifications/test_models.py`

**Step 1: Write the failing tests**

```python
# tests/notifications/test_models.py
"""Tests for notification models."""

from datetime import datetime

import pytest


class TestAlertType:
    """Tests for AlertType enum."""

    def test_new_signal_value(self):
        """AlertType.NEW_SIGNAL has correct value."""
        from notifications.models import AlertType

        assert AlertType.NEW_SIGNAL.value == "new_signal"

    def test_entry_executed_value(self):
        """AlertType.ENTRY_EXECUTED has correct value."""
        from notifications.models import AlertType

        assert AlertType.ENTRY_EXECUTED.value == "entry_executed"

    def test_exit_executed_value(self):
        """AlertType.EXIT_EXECUTED has correct value."""
        from notifications.models import AlertType

        assert AlertType.EXIT_EXECUTED.value == "exit_executed"

    def test_circuit_breaker_value(self):
        """AlertType.CIRCUIT_BREAKER has correct value."""
        from notifications.models import AlertType

        assert AlertType.CIRCUIT_BREAKER.value == "circuit_breaker"

    def test_daily_summary_value(self):
        """AlertType.DAILY_SUMMARY has correct value."""
        from notifications.models import AlertType

        assert AlertType.DAILY_SUMMARY.value == "daily_summary"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation_minimal(self):
        """Alert can be created with just alert_type."""
        from notifications.models import Alert, AlertType

        alert = Alert(alert_type=AlertType.NEW_SIGNAL)

        assert alert.alert_type == AlertType.NEW_SIGNAL
        assert alert.symbol is None
        assert alert.message == ""
        assert alert.data is None
        assert isinstance(alert.timestamp, datetime)

    def test_alert_creation_full(self):
        """Alert can be created with all fields."""
        from notifications.models import Alert, AlertType

        ts = datetime(2026, 1, 17, 10, 30, 0)
        alert = Alert(
            alert_type=AlertType.ENTRY_EXECUTED,
            symbol="AAPL",
            message="Entry executed",
            data={"price": 178.50},
            timestamp=ts,
        )

        assert alert.alert_type == AlertType.ENTRY_EXECUTED
        assert alert.symbol == "AAPL"
        assert alert.message == "Entry executed"
        assert alert.data == {"price": 178.50}
        assert alert.timestamp == ts
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/notifications/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'notifications'"

**Step 3: Create package structure**

```python
# src/notifications/__init__.py
"""Notifications module for trading alerts via Telegram."""

from .models import Alert, AlertType

__all__ = [
    "Alert",
    "AlertType",
]
```

```python
# tests/notifications/__init__.py
"""Tests for notifications module."""
```

**Step 4: Implement the models**

```python
# src/notifications/models.py
"""Data models for notifications."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertType(Enum):
    """Type of alert to send."""

    NEW_SIGNAL = "new_signal"
    ENTRY_EXECUTED = "entry_executed"
    EXIT_EXECUTED = "exit_executed"
    CIRCUIT_BREAKER = "circuit_breaker"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class Alert:
    """An alert to be sent via Telegram.

    Attributes:
        alert_type: Type of alert.
        symbol: Stock symbol (if applicable).
        message: Pre-formatted message (if not using formatter).
        data: Additional data for formatting.
        timestamp: When the alert was created.
    """

    alert_type: AlertType
    symbol: str | None = None
    message: str = ""
    data: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/notifications/test_models.py -v`
Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add src/notifications/ tests/notifications/
git commit -m "feat(notifications): add Alert and AlertType models"
```

---

## Task 2: NotificationSettings

**Files:**
- Create: `src/notifications/settings.py`
- Modify: `src/notifications/__init__.py`
- Modify: `src/config/settings.py`
- Create: `tests/notifications/test_settings.py`

**Step 1: Write the failing tests**

```python
# tests/notifications/test_settings.py
"""Tests for notification settings."""

import pytest


class TestNotificationSettings:
    """Tests for NotificationSettings."""

    def test_default_values(self):
        """Settings have correct defaults."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings()

        assert settings.enabled is True
        assert settings.telegram_token == ""
        assert settings.chat_id == ""
        assert "new_signal" in settings.alert_types
        assert "entry_executed" in settings.alert_types
        assert "circuit_breaker" in settings.alert_types
        assert settings.pre_market_checklist_enabled is True
        assert settings.pre_market_checklist_time == "09:00"
        assert len(settings.checklist_items) == 5

    def test_custom_values(self):
        """Settings accept custom values."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            enabled=False,
            telegram_token="test_token",
            chat_id="12345",
            alert_types=["new_signal"],
            pre_market_checklist_enabled=False,
        )

        assert settings.enabled is False
        assert settings.telegram_token == "test_token"
        assert settings.chat_id == "12345"
        assert settings.alert_types == ["new_signal"]
        assert settings.pre_market_checklist_enabled is False

    def test_is_configured_false_when_no_token(self):
        """is_configured returns False when token is empty."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(telegram_token="", chat_id="12345")

        assert settings.is_configured is False

    def test_is_configured_false_when_no_chat_id(self):
        """is_configured returns False when chat_id is empty."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(telegram_token="token", chat_id="")

        assert settings.is_configured is False

    def test_is_configured_true_when_both_set(self):
        """is_configured returns True when both token and chat_id are set."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(telegram_token="token", chat_id="12345")

        assert settings.is_configured is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/notifications/test_settings.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement the settings**

```python
# src/notifications/settings.py
"""Settings for notifications module."""

from pydantic import BaseModel, computed_field


class NotificationSettings(BaseModel):
    """Configuration for Telegram notifications.

    Attributes:
        enabled: Whether notifications are enabled.
        telegram_token: Bot token from BotFather.
        chat_id: Telegram chat ID to send messages to.
        alert_types: Which alert types to send.
        pre_market_checklist_enabled: Whether to send pre-market checklist.
        pre_market_checklist_time: Time to send checklist (HH:MM).
        checklist_items: Items for the pre-market checklist.
    """

    enabled: bool = True
    telegram_token: str = ""
    chat_id: str = ""

    alert_types: list[str] = [
        "new_signal",
        "entry_executed",
        "exit_executed",
        "circuit_breaker",
        "daily_summary",
    ]

    pre_market_checklist_enabled: bool = True
    pre_market_checklist_time: str = "09:00"
    checklist_items: list[str] = [
        "Economic calendar reviewed",
        "Overnight news checked",
        "Watchlist prepared",
        "Mental state: focused",
        "Risk parameters confirmed",
    ]

    @computed_field
    @property
    def is_configured(self) -> bool:
        """Check if Telegram credentials are configured."""
        return bool(self.telegram_token and self.chat_id)
```

**Step 4: Update __init__.py**

```python
# src/notifications/__init__.py
"""Notifications module for trading alerts via Telegram."""

from .models import Alert, AlertType
from .settings import NotificationSettings

__all__ = [
    "Alert",
    "AlertType",
    "NotificationSettings",
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/notifications/test_settings.py -v`
Expected: PASS (5 tests)

**Step 6: Update config/settings.py to include NotificationSettings**

Read current `src/config/settings.py` to understand structure, then add:

```python
# Add import at top
from notifications.settings import NotificationSettings

# Add field to Settings class
notifications: NotificationSettings = NotificationSettings()
```

**Step 7: Commit**

```bash
git add src/notifications/settings.py src/notifications/__init__.py src/config/settings.py tests/notifications/test_settings.py
git commit -m "feat(notifications): add NotificationSettings"
```

---

## Task 3: AlertFormatter

**Files:**
- Create: `src/notifications/alert_formatter.py`
- Modify: `src/notifications/__init__.py`
- Create: `tests/notifications/test_formatter.py`

**Step 1: Write the failing tests**

```python
# tests/notifications/test_formatter.py
"""Tests for AlertFormatter."""

from datetime import datetime

import pytest


class TestAlertFormatterNewSignal:
    """Tests for format_new_signal."""

    def test_format_long_signal(self):
        """Formats LONG signal correctly."""
        from notifications.alert_formatter import AlertFormatter
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        rec = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=5.0,
            entry_price=178.50,
            stop_loss=176.00,
            take_profit=183.00,
            risk_reward_ratio=1.8,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.1,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="High sentiment confidence",
            timestamp=datetime(2026, 1, 17, 10, 30),
        )

        formatter = AlertFormatter()
        message = formatter.format_new_signal(rec)

        assert "AAPL" in message
        assert "LONG" in message
        assert "85" in message
        assert "STRONG" in message
        assert "178.50" in message
        assert "176.00" in message
        assert "183.00" in message
        assert "1.8" in message or "1:1.8" in message
        assert "High sentiment confidence" in message

    def test_format_short_signal(self):
        """Formats SHORT signal correctly."""
        from notifications.alert_formatter import AlertFormatter
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        rec = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            score=72.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=3.0,
            entry_price=250.00,
            stop_loss=255.00,
            take_profit=240.00,
            risk_reward_ratio=2.0,
            components=ScoreComponents(
                sentiment_score=75.0,
                technical_score=70.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.05,
                credibility_multiplier=1.0,
                time_factor=0.9,
            ),
            reasoning="Bearish sentiment",
            timestamp=datetime(2026, 1, 17, 11, 0),
        )

        formatter = AlertFormatter()
        message = formatter.format_new_signal(rec)

        assert "TSLA" in message
        assert "SHORT" in message
        assert "72" in message
        assert "MODERATE" in message


class TestAlertFormatterExecution:
    """Tests for format_execution."""

    def test_format_entry_execution(self):
        """Formats entry execution correctly."""
        from execution.models import ExecutionResult
        from notifications.alert_formatter import AlertFormatter

        result = ExecutionResult(
            success=True,
            order_id="order123",
            symbol="AAPL",
            side="buy",
            quantity=50,
            filled_price=178.52,
            error_message=None,
            timestamp=datetime(2026, 1, 17, 10, 35),
        )

        formatter = AlertFormatter()
        message = formatter.format_execution(result, is_entry=True)

        assert "ENTRY" in message
        assert "AAPL" in message
        assert "50" in message
        assert "178.52" in message

    def test_format_exit_execution(self):
        """Formats exit execution correctly."""
        from execution.models import ExecutionResult
        from notifications.alert_formatter import AlertFormatter

        result = ExecutionResult(
            success=True,
            order_id="order456",
            symbol="AAPL",
            side="sell",
            quantity=50,
            filled_price=181.20,
            error_message=None,
            timestamp=datetime(2026, 1, 17, 14, 0),
        )

        formatter = AlertFormatter()
        message = formatter.format_execution(result, is_entry=False)

        assert "EXIT" in message
        assert "AAPL" in message
        assert "50" in message
        assert "181.20" in message


class TestAlertFormatterCircuitBreaker:
    """Tests for format_circuit_breaker."""

    def test_format_circuit_breaker(self):
        """Formats circuit breaker alert correctly."""
        from datetime import date

        from notifications.alert_formatter import AlertFormatter
        from risk.models import DailyRiskState

        state = DailyRiskState(
            date=date(2026, 1, 17),
            realized_pnl=-450.0,
            unrealized_pnl=0.0,
            trades_today=5,
            is_blocked=True,
        )

        formatter = AlertFormatter()
        message = formatter.format_circuit_breaker("Daily loss limit reached", state)

        assert "CIRCUIT BREAKER" in message
        assert "Daily loss limit reached" in message
        assert "450" in message
        assert "5" in message
        assert "blocked" in message.lower()


class TestAlertFormatterDailySummary:
    """Tests for format_daily_summary."""

    def test_format_daily_summary_with_trades(self):
        """Formats daily summary with trades."""
        from notifications.alert_formatter import AlertFormatter

        stats = {
            "date": "2026-01-17",
            "total_trades": 5,
            "winners": 3,
            "losers": 2,
            "gross_pnl": 285.0,
            "largest_win": 180.0,
            "largest_win_symbol": "AAPL",
            "largest_loss": -95.0,
            "largest_loss_symbol": "MSFT",
            "profit_factor": 1.8,
        }

        formatter = AlertFormatter()
        message = formatter.format_daily_summary(stats)

        assert "DAILY SUMMARY" in message
        assert "2026-01-17" in message
        assert "5" in message
        assert "3" in message or "60%" in message
        assert "285" in message
        assert "1.8" in message

    def test_format_daily_summary_no_trades(self):
        """Formats daily summary with no trades."""
        from notifications.alert_formatter import AlertFormatter

        stats = {
            "date": "2026-01-17",
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "gross_pnl": 0.0,
        }

        formatter = AlertFormatter()
        message = formatter.format_daily_summary(stats)

        assert "DAILY SUMMARY" in message
        assert "No trades" in message or "0" in message
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/notifications/test_formatter.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement AlertFormatter**

```python
# src/notifications/alert_formatter.py
"""Formats alerts for Telegram messages."""

from execution.models import ExecutionResult
from risk.models import DailyRiskState
from scoring.models import TradeRecommendation


class AlertFormatter:
    """Formats trading data into readable Telegram messages."""

    def format_new_signal(self, recommendation: TradeRecommendation) -> str:
        """Format a new trading signal alert.

        Args:
            recommendation: The trade recommendation to format.

        Returns:
            Formatted message string with emoji and markdown.
        """
        direction_emoji = "📈" if recommendation.direction.value == "long" else "📉"

        # Calculate stop/target percentages
        stop_pct = abs(
            (recommendation.stop_loss - recommendation.entry_price)
            / recommendation.entry_price
            * 100
        )
        target_pct = abs(
            (recommendation.take_profit - recommendation.entry_price)
            / recommendation.entry_price
            * 100
        )

        return f"""🚨 NEW SIGNAL: {recommendation.symbol}

{direction_emoji} Direction: {recommendation.direction.value.upper()}
⭐ Score: {recommendation.score:.0f}/100 ({recommendation.tier.value.upper()})
💰 Entry: ${recommendation.entry_price:.2f}
🛑 Stop: ${recommendation.stop_loss:.2f} (-{stop_pct:.1f}%)
🎯 Target: ${recommendation.take_profit:.2f} (+{target_pct:.1f}%)
📊 R:R: 1:{recommendation.risk_reward_ratio:.1f}

📝 {recommendation.reasoning}"""

    def format_execution(self, result: ExecutionResult, is_entry: bool) -> str:
        """Format an execution result (entry or exit).

        Args:
            result: The execution result to format.
            is_entry: True for entry, False for exit.

        Returns:
            Formatted message string.
        """
        if is_entry:
            emoji = "✅"
            title = "ENTRY EXECUTED"
            direction = "📈" if result.side == "buy" else "📉"
            action = "LONG" if result.side == "buy" else "SHORT"
        else:
            emoji = "🏁"
            title = "EXIT EXECUTED"
            direction = "📉"
            action = "SOLD" if result.side == "sell" else "COVERED"

        value = result.quantity * (result.filled_price or 0)

        return f"""{emoji} {title}

{direction} {result.symbol} - {action}
📦 Quantity: {result.quantity} shares
💵 Price: ${result.filled_price:.2f}
💰 Value: ${value:,.2f}"""

    def format_circuit_breaker(
        self, reason: str, risk_state: DailyRiskState
    ) -> str:
        """Format a circuit breaker trigger alert.

        Args:
            reason: Why the circuit breaker was triggered.
            risk_state: Current daily risk state.

        Returns:
            Formatted message string.
        """
        return f"""🚫 CIRCUIT BREAKER TRIGGERED

⚠️ Reason: {reason}
📉 Daily P&L: ${risk_state.realized_pnl:,.2f}
📊 Trades today: {risk_state.trades_today}
🔒 Trading blocked until tomorrow"""

    def format_daily_summary(self, stats: dict) -> str:
        """Format daily trading summary.

        Args:
            stats: Dictionary with daily statistics.

        Returns:
            Formatted message string.
        """
        total_trades = stats.get("total_trades", 0)

        if total_trades == 0:
            return f"""📊 DAILY SUMMARY - {stats.get('date', 'N/A')}

No trades executed today."""

        winners = stats.get("winners", 0)
        losers = stats.get("losers", 0)
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
        gross_pnl = stats.get("gross_pnl", 0)
        pnl_emoji = "📈" if gross_pnl >= 0 else "📉"

        message = f"""📊 DAILY SUMMARY - {stats.get('date', 'N/A')}

📈 Trades: {total_trades}
✅ Winners: {winners} ({win_rate:.0f}%)
❌ Losers: {losers}

{pnl_emoji} Gross P&L: ${gross_pnl:+,.2f}"""

        if stats.get("largest_win"):
            message += f"\n🏆 Largest Win: ${stats['largest_win']:+,.2f} ({stats.get('largest_win_symbol', '')})"

        if stats.get("largest_loss"):
            message += f"\n💔 Largest Loss: ${stats['largest_loss']:,.2f} ({stats.get('largest_loss_symbol', '')})"

        if stats.get("profit_factor"):
            message += f"\n\n⭐ Profit Factor: {stats['profit_factor']:.1f}"

        return message
```

**Step 4: Update __init__.py**

```python
# src/notifications/__init__.py
"""Notifications module for trading alerts via Telegram."""

from .alert_formatter import AlertFormatter
from .models import Alert, AlertType
from .settings import NotificationSettings

__all__ = [
    "Alert",
    "AlertFormatter",
    "AlertType",
    "NotificationSettings",
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/notifications/test_formatter.py -v`
Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add src/notifications/alert_formatter.py src/notifications/__init__.py tests/notifications/test_formatter.py
git commit -m "feat(notifications): add AlertFormatter"
```

---

## Task 4: TelegramNotifier Core

**Files:**
- Create: `src/notifications/telegram_notifier.py`
- Modify: `src/notifications/__init__.py`
- Create: `tests/notifications/test_notifier.py`

**Step 1: Write the failing tests**

```python
# tests/notifications/test_notifier.py
"""Tests for TelegramNotifier."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTelegramNotifierInit:
    """Tests for TelegramNotifier initialization."""

    def test_init_with_settings(self):
        """Notifier initializes with settings."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()

        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        assert notifier._settings == settings
        assert notifier._formatter == formatter
        assert notifier._bot is None

    def test_is_enabled_false_when_disabled(self):
        """is_enabled returns False when disabled in settings."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            enabled=False,
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        assert notifier.is_enabled is False

    def test_is_enabled_false_when_not_configured(self):
        """is_enabled returns False when token/chat_id missing."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            enabled=True,
            telegram_token="",
            chat_id="",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        assert notifier.is_enabled is False

    def test_is_enabled_true_when_configured(self):
        """is_enabled returns True when enabled and configured."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            enabled=True,
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        assert notifier.is_enabled is True


class TestTelegramNotifierStartStop:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_bot(self):
        """start() creates bot instance."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="test_token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        with patch("notifications.telegram_notifier.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot_class.return_value = mock_bot

            await notifier.start()

            mock_bot_class.assert_called_once_with(token="test_token")
            assert notifier._bot is not None

    @pytest.mark.asyncio
    async def test_stop_clears_bot(self):
        """stop() clears bot instance."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="test_token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())
        notifier._bot = MagicMock()

        await notifier.stop()

        assert notifier._bot is None


class TestTelegramNotifierSendAlert:
    """Tests for send_alert method."""

    @pytest.mark.asyncio
    async def test_send_alert_returns_false_when_disabled(self):
        """send_alert returns False when notifier is disabled."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.models import Alert, AlertType
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(enabled=False)
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        alert = Alert(alert_type=AlertType.NEW_SIGNAL, message="test")
        result = await notifier.send_alert(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_returns_true_on_success(self):
        """send_alert returns True when message sent successfully."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.models import Alert, AlertType
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        alert = Alert(alert_type=AlertType.NEW_SIGNAL, message="test message")
        result = await notifier.send_alert(alert)

        assert result is True
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_returns_false_on_error(self):
        """send_alert returns False on error (graceful degradation)."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.models import Alert, AlertType
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("Network error"))
        notifier._bot = mock_bot

        alert = Alert(alert_type=AlertType.NEW_SIGNAL, message="test")
        result = await notifier.send_alert(alert)

        assert result is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/notifications/test_notifier.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement TelegramNotifier**

```python
# src/notifications/telegram_notifier.py
"""Telegram notification sender."""

import logging

from telegram import Bot

from .alert_formatter import AlertFormatter
from .models import Alert
from .settings import NotificationSettings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends alerts via Telegram.

    Attributes:
        _settings: Notification settings.
        _formatter: Alert formatter.
        _bot: Telegram bot instance.
    """

    def __init__(
        self,
        settings: NotificationSettings,
        formatter: AlertFormatter,
    ):
        """Initialize the notifier.

        Args:
            settings: Notification settings.
            formatter: Alert formatter for message formatting.
        """
        self._settings = settings
        self._formatter = formatter
        self._bot: Bot | None = None

    async def start(self) -> None:
        """Initialize the Telegram bot."""
        if not self.is_enabled:
            logger.info("Telegram notifications disabled")
            return

        self._bot = Bot(token=self._settings.telegram_token)
        logger.info("Telegram notifier started")

    async def stop(self) -> None:
        """Shutdown the bot gracefully."""
        self._bot = None
        logger.info("Telegram notifier stopped")

    async def send_alert(self, alert: Alert) -> bool:
        """Send a generic alert.

        Args:
            alert: The alert to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_enabled:
            return False

        if self._bot is None:
            logger.warning("Bot not initialized, cannot send alert")
            return False

        # Check if this alert type is enabled
        if alert.alert_type.value not in self._settings.alert_types:
            logger.debug(f"Alert type {alert.alert_type.value} not enabled")
            return False

        try:
            await self._bot.send_message(
                chat_id=self._settings.chat_id,
                text=alert.message,
                parse_mode="HTML",
            )
            logger.info(f"Sent {alert.alert_type.value} alert")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled and configured."""
        return self._settings.enabled and self._settings.is_configured
```

**Step 4: Update __init__.py**

```python
# src/notifications/__init__.py
"""Notifications module for trading alerts via Telegram."""

from .alert_formatter import AlertFormatter
from .models import Alert, AlertType
from .settings import NotificationSettings
from .telegram_notifier import TelegramNotifier

__all__ = [
    "Alert",
    "AlertFormatter",
    "AlertType",
    "NotificationSettings",
    "TelegramNotifier",
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/notifications/test_notifier.py -v`
Expected: PASS (10 tests)

**Step 6: Commit**

```bash
git add src/notifications/telegram_notifier.py src/notifications/__init__.py tests/notifications/test_notifier.py
git commit -m "feat(notifications): add TelegramNotifier core"
```

---

## Task 5: TelegramNotifier Helper Methods

**Files:**
- Modify: `src/notifications/telegram_notifier.py`
- Modify: `tests/notifications/test_notifier.py`

**Step 1: Add more tests for helper methods**

```python
# Add to tests/notifications/test_notifier.py

class TestTelegramNotifierHelpers:
    """Tests for helper methods."""

    @pytest.mark.asyncio
    async def test_send_new_signal(self):
        """send_new_signal formats and sends signal."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        rec = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=5.0,
            entry_price=178.50,
            stop_loss=176.00,
            take_profit=183.00,
            risk_reward_ratio=1.8,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.1,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="High sentiment",
            timestamp=datetime.now(),
        )

        result = await notifier.send_new_signal(rec)

        assert result is True
        call_args = mock_bot.send_message.call_args
        assert "AAPL" in call_args.kwargs["text"]
        assert "LONG" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_execution_entry(self):
        """send_execution sends entry execution."""
        from execution.models import ExecutionResult
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        result = ExecutionResult(
            success=True,
            order_id="order123",
            symbol="AAPL",
            side="buy",
            quantity=50,
            filled_price=178.52,
            error_message=None,
            timestamp=datetime.now(),
        )

        sent = await notifier.send_execution(result, is_entry=True)

        assert sent is True
        call_args = mock_bot.send_message.call_args
        assert "ENTRY" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_circuit_breaker(self):
        """send_circuit_breaker sends circuit breaker alert."""
        from datetime import date

        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier
        from risk.models import DailyRiskState

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        state = DailyRiskState(
            date=date.today(),
            realized_pnl=-450.0,
            unrealized_pnl=0.0,
            trades_today=5,
            is_blocked=True,
        )

        sent = await notifier.send_circuit_breaker("Daily limit", state)

        assert sent is True
        call_args = mock_bot.send_message.call_args
        assert "CIRCUIT BREAKER" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_daily_summary(self):
        """send_daily_summary sends summary."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        stats = {
            "date": "2026-01-17",
            "total_trades": 5,
            "winners": 3,
            "losers": 2,
            "gross_pnl": 285.0,
        }

        sent = await notifier.send_daily_summary(stats)

        assert sent is True
        call_args = mock_bot.send_message.call_args
        assert "DAILY SUMMARY" in call_args.kwargs["text"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/notifications/test_notifier.py::TestTelegramNotifierHelpers -v`
Expected: FAIL with "AttributeError: 'TelegramNotifier' object has no attribute 'send_new_signal'"

**Step 3: Implement helper methods**

```python
# Add to src/notifications/telegram_notifier.py

from execution.models import ExecutionResult
from risk.models import DailyRiskState
from scoring.models import TradeRecommendation

# Add these methods to TelegramNotifier class:

    async def send_new_signal(self, recommendation: TradeRecommendation) -> bool:
        """Send a new signal alert.

        Args:
            recommendation: The trade recommendation.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_new_signal(recommendation)
        alert = Alert(
            alert_type=AlertType.NEW_SIGNAL,
            symbol=recommendation.symbol,
            message=message,
        )
        return await self.send_alert(alert)

    async def send_execution(
        self, result: ExecutionResult, is_entry: bool
    ) -> bool:
        """Send an execution alert.

        Args:
            result: The execution result.
            is_entry: True for entry, False for exit.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_execution(result, is_entry)
        alert_type = AlertType.ENTRY_EXECUTED if is_entry else AlertType.EXIT_EXECUTED
        alert = Alert(
            alert_type=alert_type,
            symbol=result.symbol,
            message=message,
        )
        return await self.send_alert(alert)

    async def send_circuit_breaker(
        self, reason: str, risk_state: DailyRiskState
    ) -> bool:
        """Send a circuit breaker alert.

        Args:
            reason: Why the circuit breaker was triggered.
            risk_state: Current risk state.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_circuit_breaker(reason, risk_state)
        alert = Alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            message=message,
        )
        return await self.send_alert(alert)

    async def send_daily_summary(self, stats: dict) -> bool:
        """Send daily trading summary.

        Args:
            stats: Dictionary with daily statistics.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_daily_summary(stats)
        alert = Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            message=message,
        )
        return await self.send_alert(alert)
```

**Step 4: Update imports in telegram_notifier.py**

```python
# Update imports at top of src/notifications/telegram_notifier.py
from execution.models import ExecutionResult
from risk.models import DailyRiskState
from scoring.models import TradeRecommendation

from .models import Alert, AlertType
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/notifications/test_notifier.py -v`
Expected: PASS (14 tests)

**Step 6: Commit**

```bash
git add src/notifications/telegram_notifier.py tests/notifications/test_notifier.py
git commit -m "feat(notifications): add TelegramNotifier helper methods"
```

---

## Task 6: ChecklistHandler

**Files:**
- Create: `src/notifications/checklist_handler.py`
- Modify: `src/notifications/__init__.py`
- Create: `tests/notifications/test_checklist.py`

**Step 1: Write the failing tests**

```python
# tests/notifications/test_checklist.py
"""Tests for ChecklistHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestChecklistHandlerInit:
    """Tests for ChecklistHandler initialization."""

    def test_init(self):
        """Handler initializes with settings."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings()
        mock_bot = MagicMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )

        assert handler._settings == settings
        assert handler._bot == mock_bot
        assert handler._chat_id == "12345"
        assert handler._checked_items == set()
        assert handler._message_id is None


class TestChecklistHandlerState:
    """Tests for checklist state management."""

    def test_is_checklist_complete_false_initially(self):
        """is_checklist_complete returns False initially."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2", "Item 3"]
        )
        handler = ChecklistHandler(
            settings=settings,
            bot=MagicMock(),
            chat_id="12345",
        )

        assert handler.is_checklist_complete() is False

    def test_is_checklist_complete_true_when_all_checked(self):
        """is_checklist_complete returns True when all items checked."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2", "Item 3"]
        )
        handler = ChecklistHandler(
            settings=settings,
            bot=MagicMock(),
            chat_id="12345",
        )
        handler._checked_items = {0, 1, 2}

        assert handler.is_checklist_complete() is True

    def test_reset_checklist_clears_state(self):
        """reset_checklist clears checked items and message ID."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings()
        handler = ChecklistHandler(
            settings=settings,
            bot=MagicMock(),
            chat_id="12345",
        )
        handler._checked_items = {0, 1, 2}
        handler._message_id = 123

        handler.reset_checklist()

        assert handler._checked_items == set()
        assert handler._message_id is None


class TestChecklistHandlerSend:
    """Tests for sending checklist."""

    @pytest.mark.asyncio
    async def test_send_checklist_sends_message_with_keyboard(self):
        """send_checklist sends message with inline keyboard."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2"]
        )
        mock_bot = MagicMock()
        mock_message = MagicMock()
        mock_message.message_id = 999
        mock_bot.send_message = AsyncMock(return_value=mock_message)

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )

        await handler.send_checklist()

        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        assert call_args.kwargs["chat_id"] == "12345"
        assert "PRE-MARKET CHECKLIST" in call_args.kwargs["text"]
        assert call_args.kwargs["reply_markup"] is not None
        assert handler._message_id == 999


class TestChecklistHandlerCheck:
    """Tests for checking items."""

    @pytest.mark.asyncio
    async def test_on_item_checked_updates_state(self):
        """on_item_checked adds item to checked set."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2"]
        )
        mock_bot = MagicMock()
        mock_bot.edit_message_text = AsyncMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )
        handler._message_id = 999

        await handler.on_item_checked(0)

        assert 0 in handler._checked_items

    @pytest.mark.asyncio
    async def test_on_item_checked_updates_message(self):
        """on_item_checked updates the message."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2"]
        )
        mock_bot = MagicMock()
        mock_bot.edit_message_text = AsyncMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )
        handler._message_id = 999

        await handler.on_item_checked(0)

        mock_bot.edit_message_text.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/notifications/test_checklist.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement ChecklistHandler**

```python
# src/notifications/checklist_handler.py
"""Pre-market checklist handler with interactive buttons."""

import logging

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from .settings import NotificationSettings

logger = logging.getLogger(__name__)


class ChecklistHandler:
    """Handles interactive pre-market checklist.

    Sends a checklist message with inline keyboard buttons.
    Users can tap buttons to mark items as checked.
    """

    def __init__(
        self,
        settings: NotificationSettings,
        bot: Bot,
        chat_id: str,
    ):
        """Initialize the checklist handler.

        Args:
            settings: Notification settings with checklist items.
            bot: Telegram bot instance.
            chat_id: Chat ID to send checklist to.
        """
        self._settings = settings
        self._bot = bot
        self._chat_id = chat_id
        self._checked_items: set[int] = set()
        self._message_id: int | None = None

    async def send_checklist(self) -> None:
        """Send the pre-market checklist with inline buttons."""
        text = self._format_checklist_message()
        keyboard = self._build_keyboard()

        message = await self._bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )
        self._message_id = message.message_id
        logger.info("Pre-market checklist sent")

    async def on_item_checked(self, item_index: int) -> None:
        """Handle when user checks an item.

        Args:
            item_index: Index of the item that was checked.
        """
        self._checked_items.add(item_index)

        if self._message_id is None:
            return

        text = self._format_checklist_message()
        keyboard = self._build_keyboard()

        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            logger.info(f"Checklist item {item_index} checked")
        except Exception as e:
            logger.error(f"Failed to update checklist: {e}")

    def is_checklist_complete(self) -> bool:
        """Check if all items have been checked.

        Returns:
            True if all items are checked.
        """
        return len(self._checked_items) >= len(self._settings.checklist_items)

    def reset_checklist(self) -> None:
        """Reset checklist for next day."""
        self._checked_items = set()
        self._message_id = None
        logger.info("Checklist reset")

    def _build_keyboard(self) -> InlineKeyboardMarkup:
        """Build inline keyboard with current checked states.

        Returns:
            InlineKeyboardMarkup with buttons for each item.
        """
        buttons = []
        for i, item in enumerate(self._settings.checklist_items):
            if i in self._checked_items:
                # Item is checked - show checkmark
                text = f"✅ {i + 1}"
            else:
                # Item not checked - show number
                text = f"☐ {i + 1}"

            buttons.append(
                InlineKeyboardButton(
                    text=text,
                    callback_data=f"checklist_{i}",
                )
            )

        # Arrange buttons in rows of 3
        rows = []
        for i in range(0, len(buttons), 3):
            rows.append(buttons[i : i + 3])

        return InlineKeyboardMarkup(rows)

    def _format_checklist_message(self) -> str:
        """Format the checklist message showing checked/unchecked items.

        Returns:
            Formatted message string.
        """
        if self.is_checklist_complete():
            return """✅ <b>CHECKLIST COMPLETE</b>

All items verified. Ready to trade!"""

        lines = ["📋 <b>PRE-MARKET CHECKLIST</b>", ""]
        lines.append("Ready for trading day? Complete all items:")
        lines.append("")

        for i, item in enumerate(self._settings.checklist_items):
            if i in self._checked_items:
                lines.append(f"✅ {item}")
            else:
                lines.append(f"☐ {item}")

        return "\n".join(lines)
```

**Step 4: Update __init__.py**

```python
# src/notifications/__init__.py
"""Notifications module for trading alerts via Telegram."""

from .alert_formatter import AlertFormatter
from .checklist_handler import ChecklistHandler
from .models import Alert, AlertType
from .settings import NotificationSettings
from .telegram_notifier import TelegramNotifier

__all__ = [
    "Alert",
    "AlertFormatter",
    "AlertType",
    "ChecklistHandler",
    "NotificationSettings",
    "TelegramNotifier",
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/notifications/test_checklist.py -v`
Expected: PASS (8 tests)

**Step 6: Commit**

```bash
git add src/notifications/checklist_handler.py src/notifications/__init__.py tests/notifications/test_checklist.py
git commit -m "feat(notifications): add ChecklistHandler"
```

---

## Task 7: Update Config Settings

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`

**Step 1: Read current config/settings.py**

Read `src/config/settings.py` to understand current structure.

**Step 2: Add NotificationSettings import and field**

Add to imports:
```python
from notifications.settings import NotificationSettings
```

Add to Settings class:
```python
notifications: NotificationSettings = NotificationSettings()
```

**Step 3: Update config/settings.yaml**

Add notifications section:
```yaml
# Phase 9: Notifications
notifications:
  enabled: true
  # telegram_token and chat_id from environment variables:
  # TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
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

**Step 4: Run existing tests to verify no regression**

Run: `pytest tests/config/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config/settings.py config/settings.yaml
git commit -m "feat(config): add notification settings"
```

---

## Task 8: Integration Tests

**Files:**
- Create: `tests/notifications/test_integration.py`

**Step 1: Write integration tests**

```python
# tests/notifications/test_integration.py
"""Integration tests for notifications module."""

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNotificationIntegration:
    """Integration tests for full notification flow."""

    @pytest.mark.asyncio
    async def test_full_signal_notification_flow(self):
        """Test complete flow: signal -> format -> send."""
        from notifications import (
            AlertFormatter,
            NotificationSettings,
            TelegramNotifier,
        )
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        # Setup
        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        # Create recommendation
        rec = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=5.0,
            entry_price=178.50,
            stop_loss=176.00,
            take_profit=183.00,
            risk_reward_ratio=1.8,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.1,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="High sentiment confidence",
            timestamp=datetime.now(),
        )

        # Send notification
        result = await notifier.send_new_signal(rec)

        # Verify
        assert result is True
        mock_bot.send_message.assert_called_once()
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "AAPL" in sent_text
        assert "LONG" in sent_text
        assert "85" in sent_text
        assert "178.50" in sent_text

    @pytest.mark.asyncio
    async def test_circuit_breaker_notification_flow(self):
        """Test circuit breaker notification flow."""
        from notifications import (
            AlertFormatter,
            NotificationSettings,
            TelegramNotifier,
        )
        from risk.models import DailyRiskState

        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        state = DailyRiskState(
            date=date.today(),
            realized_pnl=-450.0,
            unrealized_pnl=0.0,
            trades_today=5,
            is_blocked=True,
        )

        result = await notifier.send_circuit_breaker("Daily loss limit", state)

        assert result is True
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "CIRCUIT BREAKER" in sent_text
        assert "450" in sent_text

    @pytest.mark.asyncio
    async def test_disabled_alert_type_not_sent(self):
        """Alert types not in settings.alert_types are not sent."""
        from notifications import (
            AlertFormatter,
            AlertType,
            Alert,
            NotificationSettings,
            TelegramNotifier,
        )

        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
            alert_types=["new_signal"],  # Only new_signal enabled
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        # Try to send circuit breaker (not enabled)
        alert = Alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            message="Test circuit breaker",
        )
        result = await notifier.send_alert(alert)

        # Should return False and not send
        assert result is False
        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_checklist_flow(self):
        """Test checklist send -> check items -> complete flow."""
        from notifications import ChecklistHandler, NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2", "Item 3"]
        )

        mock_bot = MagicMock()
        mock_message = MagicMock()
        mock_message.message_id = 999
        mock_bot.send_message = AsyncMock(return_value=mock_message)
        mock_bot.edit_message_text = AsyncMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )

        # Send checklist
        await handler.send_checklist()
        assert handler._message_id == 999
        assert handler.is_checklist_complete() is False

        # Check items one by one
        await handler.on_item_checked(0)
        assert handler.is_checklist_complete() is False

        await handler.on_item_checked(1)
        assert handler.is_checklist_complete() is False

        await handler.on_item_checked(2)
        assert handler.is_checklist_complete() is True

        # Verify edit_message_text was called for each check
        assert mock_bot.edit_message_text.call_count == 3

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_telegram_error(self):
        """Notifier handles Telegram errors gracefully."""
        from notifications import (
            AlertFormatter,
            AlertType,
            Alert,
            NotificationSettings,
            TelegramNotifier,
        )

        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(
            side_effect=Exception("Telegram API error")
        )
        notifier._bot = mock_bot

        alert = Alert(
            alert_type=AlertType.NEW_SIGNAL,
            message="Test message",
        )

        # Should not raise, just return False
        result = await notifier.send_alert(alert)
        assert result is False
```

**Step 2: Run integration tests**

Run: `pytest tests/notifications/test_integration.py -v`
Expected: PASS (5 tests)

**Step 3: Commit**

```bash
git add tests/notifications/test_integration.py
git commit -m "test(notifications): add integration tests"
```

---

## Task 9: Final Test Suite Run

**Step 1: Run all notification tests**

Run: `pytest tests/notifications/ -v`
Expected: All tests pass (30+ tests)

**Step 2: Run full test suite**

Run: `pytest --tb=short`
Expected: All tests pass, including previous phases

**Step 3: Check coverage**

Run: `pytest --cov=src/notifications --cov-report=term-missing tests/notifications/`
Expected: 90%+ coverage

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(notifications): Phase 9 complete - Telegram notifications"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | AlertType, Alert models | 7 |
| 2 | NotificationSettings | 5 |
| 3 | AlertFormatter | 7 |
| 4 | TelegramNotifier core | 10 |
| 5 | TelegramNotifier helpers | 4 |
| 6 | ChecklistHandler | 8 |
| 7 | Config integration | 0 (regression) |
| 8 | Integration tests | 5 |
| 9 | Final verification | 0 |

**Total estimated tests: ~46**
