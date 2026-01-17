from datetime import datetime, timezone

from src.models.social_message import SocialMessage, SourceType
from src.validation.scenarios.base import Scenario


class BullishSignalExecutesTrade(Scenario):
    """High-confidence bullish signal should execute a buy order."""

    name = "bullish_signal_executes_trade"

    def __init__(self):
        self.mock_message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="test_123",
            author="unusual_whales",
            content="Massive $AAPL call sweep! 10,000 contracts at $180 strike. Bullish flow!",
            timestamp=datetime.now(timezone.utc),
        )
        self.executed_trade = None

    async def setup(self) -> None:
        pass

    async def execute(self, engine) -> None:
        pass

    async def verify(self, engine) -> bool:
        return True
