# src/scoring/source_credibility.py
"""Source credibility management for scoring system."""
from typing import Optional

from src.models.social_message import SourceType


# Default tier 1 sources (premium/trusted sources)
DEFAULT_TIER1_SOURCES = [
    "unusual_whales",
    "optionsflow",
    "wallstreetbets",  # moderators only
    "thestreet",
    "zaborskierik",
]


class SourceCredibilityManager:
    """Maps authors to credibility tiers and multipliers.

    Tier 1: Premium/trusted sources with higher multiplier
    Tier 2: Manually promoted sources (reserved for future use)
    Tier 3: Unknown/default sources with lower multiplier

    Attributes:
        tier1_sources: Set of author names in tier 1.
        tier1_multiplier: Multiplier for tier 1 sources.
        tier2_multiplier: Multiplier for tier 2 sources (future use).
        tier3_multiplier: Multiplier for tier 3 sources.
    """

    def __init__(
        self,
        tier1_sources: Optional[list[str]] = None,
        tier1_multiplier: float = 1.2,
        tier2_multiplier: float = 1.0,
        tier3_multiplier: float = 0.8,
    ):
        """Initialize the credibility manager.

        Args:
            tier1_sources: List of premium sources. If None, uses defaults.
            tier1_multiplier: Multiplier for tier 1 sources (default 1.2).
            tier2_multiplier: Multiplier for tier 2 sources (default 1.0).
            tier3_multiplier: Multiplier for tier 3 sources (default 0.8).
        """
        if tier1_sources is not None:
            self._tier1_sources: set[str] = set(tier1_sources)
        else:
            self._tier1_sources = set(DEFAULT_TIER1_SOURCES)

        self._tier1_multiplier = tier1_multiplier
        self._tier2_multiplier = tier2_multiplier
        self._tier3_multiplier = tier3_multiplier

    def get_multiplier(
        self, author: str, source: Optional[SourceType] = None
    ) -> float:
        """Get credibility multiplier for an author.

        Args:
            author: The author/username to look up.
            source: Optional source type (for future extensibility).

        Returns:
            Credibility multiplier based on author's tier.
        """
        tier = self.get_tier(author)

        if tier == 1:
            return self._tier1_multiplier
        elif tier == 2:
            return self._tier2_multiplier
        else:
            return self._tier3_multiplier

    def get_tier(self, author: str) -> int:
        """Get tier (1, 2, or 3) for an author.

        Args:
            author: The author/username to look up.

        Returns:
            Tier number (1, 2, or 3).
        """
        if author in self._tier1_sources:
            return 1
        # Tier 2 reserved for future use (manually promoted sources)
        return 3

    def add_tier1_source(self, author: str) -> None:
        """Dynamically add a trusted source to tier 1.

        Args:
            author: The author/username to add to tier 1.
        """
        self._tier1_sources.add(author)
