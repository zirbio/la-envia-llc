# tests/research/models/test_trading_idea.py

import pytest
from src.research.models.trading_idea import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
)


class TestEnums:
    def test_direction_values(self):
        assert Direction.LONG == "LONG"
        assert Direction.SHORT == "SHORT"

    def test_conviction_values(self):
        assert Conviction.HIGH == "HIGH"
        assert Conviction.MEDIUM == "MEDIUM"
        assert Conviction.LOW == "LOW"

    def test_position_size_values(self):
        assert PositionSize.FULL == "FULL"
        assert PositionSize.HALF == "HALF"
        assert PositionSize.QUARTER == "QUARTER"


class TestTechnicalLevels:
    def test_create_technical_levels(self):
        levels = TechnicalLevels(
            support=135.50,
            resistance=142.00,
            entry_zone=(136.00, 137.50),
        )
        assert levels.support == 135.50
        assert levels.resistance == 142.00
        assert levels.entry_zone == (136.00, 137.50)


class TestRiskReward:
    def test_create_risk_reward(self):
        rr = RiskReward(
            entry=137.00,
            stop=134.50,
            target=145.00,
            ratio="3.2:1",
        )
        assert rr.entry == 137.00
        assert rr.stop == 134.50
        assert rr.target == 145.00
        assert rr.ratio == "3.2:1"


class TestTradingIdea:
    def test_create_trading_idea(self):
        idea = TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="TSMC beat + AI guidance raise",
            thesis="Semiconductor demand exceeding expectations",
            technical=TechnicalLevels(
                support=135.50,
                resistance=142.00,
                entry_zone=(136.00, 137.50),
            ),
            risk_reward=RiskReward(
                entry=137.00,
                stop=134.50,
                target=145.00,
                ratio="3.2:1",
            ),
            position_size=PositionSize.FULL,
            kill_switch="China export restrictions headline",
        )
        assert idea.ticker == "NVDA"
        assert idea.direction == Direction.LONG
        assert idea.conviction == Conviction.HIGH

    def test_trading_idea_json_serialization(self):
        idea = TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="Test catalyst",
            thesis="Test thesis",
            technical=TechnicalLevels(
                support=135.50,
                resistance=142.00,
                entry_zone=(136.00, 137.50),
            ),
            risk_reward=RiskReward(
                entry=137.00,
                stop=134.50,
                target=145.00,
                ratio="3.2:1",
            ),
            position_size=PositionSize.FULL,
            kill_switch="Test kill switch",
        )
        json_data = idea.model_dump_json()
        assert "NVDA" in json_data
        assert "LONG" in json_data
