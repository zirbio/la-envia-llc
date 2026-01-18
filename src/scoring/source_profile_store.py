# src/scoring/source_profile_store.py
"""Persistence layer for source profiles."""

import json
from datetime import datetime
from pathlib import Path

from src.models.social_message import SourceType
from src.scoring.source_profile import SourceProfile


class SourceProfileStore:
    """Stores and retrieves source profiles from disk.

    Uses JSON files for persistence with an in-memory cache for fast lookups.
    Each profile is stored as a separate file named {author_id}.json.
    """

    def __init__(self, data_dir: Path = Path("data/sources")):
        """Initialize the store.

        Args:
            data_dir: Directory to store profile JSON files.
        """
        self._data_dir = data_dir
        self._cache: dict[str, SourceProfile] = {}
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def get(self, author_id: str) -> SourceProfile | None:
        """Get profile by author_id.

        Args:
            author_id: Unique identifier for the author.

        Returns:
            SourceProfile if found, None otherwise.
        """
        # Check cache first
        if author_id in self._cache:
            return self._cache[author_id]

        # Try loading from disk
        file_path = self._data_dir / f"{author_id}.json"
        if file_path.exists():
            profile = self._load_from_file(file_path)
            self._cache[author_id] = profile
            return profile

        return None

    def save(self, profile: SourceProfile) -> None:
        """Save or update a profile.

        Args:
            profile: SourceProfile to save.
        """
        self._cache[profile.author_id] = profile
        file_path = self._data_dir / f"{profile.author_id}.json"
        self._save_to_file(profile, file_path)

    def get_or_create(
        self, author_id: str, source_type: SourceType
    ) -> SourceProfile:
        """Get existing profile or create a new one.

        Args:
            author_id: Unique identifier for the author.
            source_type: Type of source (GROK, TWITTER, etc.).

        Returns:
            Existing or newly created SourceProfile.
        """
        profile = self.get(author_id)
        if profile is None:
            now = datetime.now()
            profile = SourceProfile(
                author_id=author_id,
                source_type=source_type,
                first_seen=now,
                last_seen=now,
            )
            self.save(profile)
        return profile

    def _load_from_file(self, file_path: Path) -> SourceProfile:
        """Load profile from JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            SourceProfile loaded from file.
        """
        with open(file_path) as f:
            data = json.load(f)

        return SourceProfile(
            author_id=data["author_id"],
            source_type=SourceType(data["source_type"]),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            total_signals=data.get("total_signals", 0),
            correct_signals=data.get("correct_signals", 0),
            followers=data.get("followers"),
            verified=data.get("verified", False),
            account_age_days=data.get("account_age_days"),
            category=data.get("category", "unknown"),
            signals_history=data.get("signals_history", []),
        )

    def _save_to_file(self, profile: SourceProfile, file_path: Path) -> None:
        """Save profile to JSON file.

        Args:
            profile: SourceProfile to save.
            file_path: Path to the JSON file.
        """
        data = {
            "author_id": profile.author_id,
            "source_type": profile.source_type.value,
            "first_seen": profile.first_seen.isoformat(),
            "last_seen": profile.last_seen.isoformat(),
            "total_signals": profile.total_signals,
            "correct_signals": profile.correct_signals,
            "accuracy": profile.accuracy,
            "credibility_multiplier": profile.credibility_multiplier,
            "followers": profile.followers,
            "verified": profile.verified,
            "account_age_days": profile.account_age_days,
            "category": profile.category,
            "signals_history": profile.signals_history,
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
