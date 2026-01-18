# tests/research/models/test_daily_brief.py

import pytest
from datetime import datetime
from src.research.models.daily_brief import (
    MarketRegime,
    WatchlistItem,
    DailyBrief,
)
from src.research.models.trading_idea import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
)


class TestMarketRegime:
    def test_create_market_regime(self):
        regime = MarketRegime(
            state="risk-on",
            trend="bullish",
            summary="Markets trending higher on strong earnings",
        )
        assert regime.state == "risk-on"
        assert regime.trend == "bullish"


class TestWatchlistItem:
    def test_create_watchlist_item(self):
        item = WatchlistItem(
            ticker="MSFT",
            setup="Consolidating near highs",
            trigger="Break above $420 with volume",
        )
        assert item.ticker == "MSFT"


class TestDailyBrief:
    def test_create_daily_brief(self):
        idea = TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="Test",
            thesis="Test",
            technical=TechnicalLevels(
                support=135.0, resistance=145.0, entry_zone=(136.0, 138.0)
            ),
            risk_reward=RiskReward(
                entry=137.0, stop=134.0, target=145.0, ratio="2.7:1"
            ),
            position_size=PositionSize.FULL,
            kill_switch="Test",
        )

        brief = DailyBrief(
            generated_at=datetime(2026, 1, 18, 12, 0, 0),
            brief_type="initial",
            market_regime=MarketRegime(
                state="risk-on",
                trend="bullish",
                summary="Test summary",
            ),
            ideas=[idea],
            watchlist=[
                WatchlistItem(ticker="MSFT", setup="Test", trigger="Test")
            ],
            risks=["CPI release at 14:30"],
            key_questions=["Will NVDA hold $135 support?"],
            data_sources_used=["grok", "sec", "yahoo"],
            fetch_duration_seconds=5.2,
            analysis_duration_seconds=12.3,
        )

        assert brief.brief_type == "initial"
        assert len(brief.ideas) == 1
        assert brief.ideas[0].ticker == "NVDA"

    def test_daily_brief_json_serialization(self):
        brief = DailyBrief(
            generated_at=datetime(2026, 1, 18, 12, 0, 0),
            brief_type="pre_open",
            market_regime=MarketRegime(
                state="neutral",
                trend="ranging",
                summary="Test",
            ),
            ideas=[],
            watchlist=[],
            risks=[],
            key_questions=[],
            data_sources_used=[],
            fetch_duration_seconds=0.0,
            analysis_duration_seconds=0.0,
        )
        json_data = brief.model_dump_json()
        assert "pre_open" in json_data
