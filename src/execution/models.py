# src/execution/models.py
"""Data models for the execution system."""
from dataclasses import dataclass
from datetime import datetime

from src.scoring.models import Direction


@dataclass
class ExecutionResult:
    """Result of an order execution attempt.

    Attributes:
        success: Did the order submit successfully?
        order_id: Alpaca order ID (if success).
        symbol: Stock symbol.
        side: "buy" or "sell".
        quantity: Shares executed.
        filled_price: Fill price (if filled).
        error_message: Error details (if failed).
        timestamp: When execution was attempted.
    """

    success: bool
    order_id: str | None
    symbol: str
    side: str
    quantity: int
    filled_price: float | None
    error_message: str | None
    timestamp: datetime


@dataclass
class TrackedPosition:
    """A position being tracked by the system.

    Attributes:
        symbol: Stock symbol.
        quantity: Shares held.
        entry_price: Average entry price.
        entry_time: When position was opened.
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        order_id: Original order ID.
        direction: LONG or SHORT.
    """

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    order_id: str
    direction: Direction
