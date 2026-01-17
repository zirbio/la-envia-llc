# src/analyzers/claude_result.py
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CatalystType(str, Enum):
    """Types of catalysts detected in social messages."""
    INSTITUTIONAL_FLOW = "institutional_flow"
    EARNINGS = "earnings"
    BREAKING_NEWS = "breaking_news"
    TECHNICAL_BREAKOUT = "technical_breakout"
    SECTOR_ROTATION = "sector_rotation"
    INSIDER_ACTIVITY = "insider_activity"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    ANALYST_UPGRADE = "analyst_upgrade"
    SHORT_SQUEEZE = "short_squeeze"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk levels for trading opportunities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ClaudeAnalysisResult(BaseModel):
    """Result from Claude API deep analysis."""

    catalyst_type: CatalystType
    catalyst_confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)
    context_summary: str
    recommendation: str
    reasoning: Optional[str] = None

    def is_actionable(self, min_confidence: float = 0.7) -> bool:
        """Check if this analysis suggests an actionable opportunity."""
        return (
            self.catalyst_confidence >= min_confidence
            and self.risk_level != RiskLevel.EXTREME
        )
