# src/notifications/alert_formatter.py
"""Formats alerts for Telegram messages."""

from execution.models import ExecutionResult
from risk.models import DailyRiskState
from scoring.models import TradeRecommendation


class AlertFormatter:
    """Formats trading data into readable Telegram messages."""

    def format_new_signal(self, recommendation: TradeRecommendation) -> str:
        """Format a new trading signal alert."""
        direction_emoji = "📈" if recommendation.direction.value == "long" else "📉"

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
        """Format an execution result (entry or exit)."""
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

    def format_circuit_breaker(self, reason: str, risk_state: DailyRiskState) -> str:
        """Format a circuit breaker trigger alert."""
        return f"""🚫 CIRCUIT BREAKER TRIGGERED

⚠️ Reason: {reason}
📉 Daily P&L: ${risk_state.realized_pnl:,.2f}
📊 Trades today: {risk_state.trades_today}
🔒 Trading blocked until tomorrow"""

    def format_daily_summary(self, stats: dict) -> str:
        """Format daily trading summary."""
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
