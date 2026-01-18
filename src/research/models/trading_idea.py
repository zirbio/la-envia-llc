"""Trading idea models for Morning Research Agent."""

from enum import Enum
from pydantic import BaseModel


class Direction(str, Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class Conviction(str, Enum):
    """Conviction level for trading idea."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PositionSize(str, Enum):
    """Recommended position size."""
    FULL = "FULL"        # 1.0x normal size
    HALF = "HALF"        # 0.5x
    QUARTER = "QUARTER"  # 0.25x


class TechnicalLevels(BaseModel):
    """Key technical levels for the trade."""
    support: float
    resistance: float
    entry_zone: tuple[float, float]


class RiskReward(BaseModel):
    """Risk/reward parameters for the trade."""
    entry: float
    stop: float
    target: float
    ratio: str  # e.g., "2.5:1"


class TradingIdea(BaseModel):
    """A single trading idea from the Daily Brief."""
    rank: int
    ticker: str
    direction: Direction
    conviction: Conviction
    catalyst: str
    thesis: str
    technical: TechnicalLevels
    risk_reward: RiskReward
    position_size: PositionSize
    kill_switch: str
