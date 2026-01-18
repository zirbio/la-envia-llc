# src/scoring/dynamic_credibility_manager.py
"""Dynamic credibility manager that learns from signal outcomes."""

from datetime import datetime

from src.models.social_message import SourceType
from src.scoring.source_profile_store import SourceProfileStore


class DynamicCredibilityManager:
    """Manages source credibility based on historical accuracy.

    Replaces static SourceCredibilityManager with dynamic learning.
    """

    def __init__(
        self,
        profile_store: SourceProfileStore,
        min_signals_for_ranking: int = 5,
        tier1_sources: list[str] | None = None,
        tier1_multiplier: float = 1.3,
    ):
        """Initialize the manager.

        Args:
            profile_store: Store for source profiles.
            min_signals_for_ranking: Minimum signals before using dynamic multiplier.
            tier1_sources: Seed list of known reliable sources.
            tier1_multiplier: Multiplier for tier1 sources without history.
        """
        self._store = profile_store
        self._min_signals = min_signals_for_ranking
        self._tier1_sources = tier1_sources or []
        self._tier1_multiplier = tier1_multiplier

    def get_multiplier(self, author_id: str, source_type: SourceType) -> float:
        """Get credibility multiplier for an author.

        Priority:
        1. If sufficient history -> use dynamic accuracy-based multiplier
        2. If in tier1_sources (seed) -> use tier1_multiplier
        3. Default -> 1.0

        Args:
            author_id: Unique identifier for the author.
            source_type: Type of source.

        Returns:
            Credibility multiplier (0.5 to 1.3).
        """
        profile = self._store.get(author_id)

        if profile and profile.total_signals >= self._min_signals:
            # Sufficient history -> use dynamic multiplier
            return profile.credibility_multiplier

        if author_id in self._tier1_sources:
            # Known source without sufficient history -> use seed
            return self._tier1_multiplier

        # Unknown author
        return 1.0

    def record_signal(
        self, author_id: str, source_type: SourceType, signal_id: str
    ) -> None:
        """Record that an author generated a signal.

        Args:
            author_id: Unique identifier for the author.
            source_type: Type of source.
            signal_id: Unique identifier for the signal.
        """
        profile = self._store.get_or_create(author_id, source_type)
        profile.total_signals += 1
        profile.last_seen = datetime.now()
        profile.signals_history.append(signal_id)
        self._store.save(profile)

    def record_outcome(self, author_id: str, was_correct: bool) -> None:
        """Record the outcome of a signal (correct/incorrect).

        Args:
            author_id: Unique identifier for the author.
            was_correct: Whether the signal prediction was correct.
        """
        profile = self._store.get(author_id)
        if profile:
            if was_correct:
                profile.correct_signals += 1
            # Accuracy is computed property, no need to update
            self._store.save(profile)
