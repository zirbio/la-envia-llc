# src/research/models/daily_brief.py

"""Daily Brief model for Morning Research Agent."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel

from src.research.models.trading_idea import TradingIdea


class MarketRegime(BaseModel):
    """Current market regime assessment."""
    state: Literal["risk-on", "risk-off", "neutral"]
    trend: Literal["bullish", "bearish", "ranging"]
    summary: str


class WatchlistItem(BaseModel):
    """A ticker to watch but not trade today."""
    ticker: str
    setup: str
    trigger: str


class DailyBrief(BaseModel):
    """The complete Daily Brief from Morning Research Agent."""
    generated_at: datetime
    brief_type: Literal["initial", "pre_open"]
    timezone: str = "Europe/Madrid"

    market_regime: MarketRegime
    ideas: list[TradingIdea]
    watchlist: list[WatchlistItem]
    risks: list[str]
    key_questions: list[str]

    # Metadata
    data_sources_used: list[str]
    fetch_duration_seconds: float
    analysis_duration_seconds: float
