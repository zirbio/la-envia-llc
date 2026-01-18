# tests/research/test_integration.py

import pytest
from datetime import datetime
from src.research.integration import idea_to_social_message
from src.research.models import (
    TradingIdea,
    DailyBrief,
    MarketRegime,
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
)
from src.models.social_message import SourceType


class TestIdeaToSocialMessage:
    @pytest.fixture
    def sample_idea(self):
        return TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="TSMC beat + AI guidance",
            thesis="Strong demand",
            technical=TechnicalLevels(
                support=135.0, resistance=145.0, entry_zone=(136.0, 138.0)
            ),
            risk_reward=RiskReward(
                entry=137.0, stop=134.0, target=145.0, ratio="2.7:1"
            ),
            position_size=PositionSize.FULL,
            kill_switch="China export ban",
        )

    @pytest.fixture
    def sample_brief(self, sample_idea):
        return DailyBrief(
            generated_at=datetime(2026, 1, 18, 12, 0, 0),
            brief_type="initial",
            market_regime=MarketRegime(
                state="risk-on", trend="bullish", summary="Test"
            ),
            ideas=[sample_idea],
            watchlist=[],
            risks=[],
            key_questions=[],
            data_sources_used=["grok"],
            fetch_duration_seconds=1.0,
            analysis_duration_seconds=2.0,
        )

    def test_converts_to_social_message(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert msg.source == SourceType.RESEARCH
        assert "NVDA" in msg.content
        assert "LONG" in msg.content
        assert msg.author == "morning_research_agent"

    def test_source_id_is_unique(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert "brief_" in msg.source_id
        assert "NVDA" in msg.source_id

    def test_metadata_includes_trade_params(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert msg.metadata["conviction"] == "HIGH"
        assert msg.metadata["direction"] == "LONG"
        assert msg.metadata["entry"] == 137.0
        assert msg.metadata["stop"] == 134.0
        assert msg.metadata["target"] == 145.0

    def test_content_includes_risk_reward(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert "Entry: $137.0" in msg.content
        assert "Stop: $134.0" in msg.content
        assert "Target: $145.0" in msg.content
        assert "2.7:1" in msg.content
