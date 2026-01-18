# src/research/integration.py

"""Integration with existing trading system."""

from src.models.social_message import SocialMessage, SourceType
from src.research.models import TradingIdea, DailyBrief


def idea_to_social_message(idea: TradingIdea, brief: DailyBrief) -> SocialMessage:
    """Convert a TradingIdea to SocialMessage for the trading system.

    Args:
        idea: The trading idea to convert.
        brief: The parent DailyBrief for context.

    Returns:
        SocialMessage that can be processed by the trading system.
    """
    content = (
        f"${idea.ticker} {idea.direction.value} - {idea.conviction.value} conviction. "
        f"Catalyst: {idea.catalyst}. "
        f"Entry: ${idea.risk_reward.entry}, Stop: ${idea.risk_reward.stop}, "
        f"Target: ${idea.risk_reward.target} (R:R {idea.risk_reward.ratio}). "
        f"Kill switch: {idea.kill_switch}"
    )

    return SocialMessage(
        source=SourceType.RESEARCH,
        source_id=f"brief_{brief.generated_at.isoformat()}_{idea.ticker}",
        author="morning_research_agent",
        content=content,
        timestamp=brief.generated_at,
        url=None,
        metadata={
            "rank": idea.rank,
            "conviction": idea.conviction.value,
            "direction": idea.direction.value,
            "entry": idea.risk_reward.entry,
            "stop": idea.risk_reward.stop,
            "target": idea.risk_reward.target,
            "position_size": idea.position_size.value,
        },
    )
