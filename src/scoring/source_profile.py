# src/scoring/source_profile.py
"""Source profile model for tracking author credibility."""

from dataclasses import dataclass, field
from datetime import datetime

from src.models.social_message import SourceType


@dataclass
class SourceProfile:
    """Profile of a source/author that generates signals.

    Tracks historical accuracy to compute dynamic credibility multiplier.

    Attributes:
        author_id: Unique identifier (e.g., Twitter username).
        source_type: Where signals come from (TWITTER, GROK, etc.).
        first_seen: When this author was first observed.
        last_seen: When this author was last observed.
        total_signals: Total number of signals from this author.
        correct_signals: Number of signals that were correct.
        followers: Author's follower count (optional).
        verified: Whether the author is verified (optional).
        account_age_days: Age of the account in days (optional).
        category: Category classification for the author.
        signals_history: List of signal IDs for history tracking.
    """

    author_id: str
    source_type: SourceType
    first_seen: datetime
    last_seen: datetime

    # Credibility metrics
    total_signals: int = 0
    correct_signals: int = 0

    # Metadata
    followers: int | None = None
    verified: bool = False
    account_age_days: int | None = None
    category: str = "unknown"

    # Signal history (IDs)
    signals_history: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy ratio.

        Returns:
            Ratio of correct signals to total signals (0.0 to 1.0).
            Returns 0.0 if no signals have been recorded.
        """
        if self.total_signals == 0:
            return 0.0
        return self.correct_signals / self.total_signals

    @property
    def credibility_multiplier(self) -> float:
        """Calculate dynamic credibility multiplier based on accuracy.

        The multiplier adjusts signal scores based on historical accuracy:
        - >= 75% accuracy: 1.3x (Tier 1+, highly reliable)
        - >= 60% accuracy: 1.1x (Tier 1, reliable)
        - >= 50% accuracy: 1.0x (Tier 2, neutral)
        - >= 40% accuracy: 0.8x (Tier 3, below average)
        - < 40% accuracy: 0.5x (Unreliable)

        Returns:
            Multiplier from 0.5 to 1.3 based on historical accuracy.
            Returns 1.0 if insufficient data (< 5 signals).
        """
        if self.total_signals < 5:
            return 1.0  # Insufficient data

        if self.accuracy >= 0.75:
            return 1.3  # Tier 1+
        elif self.accuracy >= 0.60:
            return 1.1  # Tier 1
        elif self.accuracy >= 0.50:
            return 1.0  # Tier 2
        elif self.accuracy >= 0.40:
            return 0.8  # Tier 3
        else:
            return 0.5  # Unreliable
