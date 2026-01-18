# tests/scoring/test_source_profile_store.py
"""Tests for SourceProfileStore."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest


def test_store_save_and_get():
    """Test saving and retrieving a profile."""
    from src.models.social_message import SourceType
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        profile = SourceProfile(
            author_id="test_user",
            source_type=SourceType.GROK,
            total_signals=10,
            correct_signals=8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        store.save(profile)
        retrieved = store.get("test_user")

        assert retrieved is not None
        assert retrieved.author_id == "test_user"
        assert retrieved.total_signals == 10
        assert retrieved.correct_signals == 8


def test_store_get_nonexistent():
    """Test getting a nonexistent profile returns None."""
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        result = store.get("nonexistent")
        assert result is None


def test_store_get_or_create():
    """Test get_or_create creates new profile and retrieves existing."""
    from src.models.social_message import SourceType
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # First call creates
        profile1 = store.get_or_create("new_user", SourceType.GROK)
        assert profile1.author_id == "new_user"
        assert profile1.total_signals == 0

        # Modify and save
        profile1.total_signals = 5
        store.save(profile1)

        # Second call retrieves existing
        profile2 = store.get_or_create("new_user", SourceType.GROK)
        assert profile2.total_signals == 5


def test_store_persistence():
    """Test data persists across store instances."""
    from src.models.social_message import SourceType
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save with first store instance
        store1 = SourceProfileStore(data_dir=Path(tmpdir))
        profile = SourceProfile(
            author_id="persistent_user",
            source_type=SourceType.GROK,
            total_signals=15,
            correct_signals=12,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store1.save(profile)

        # Create new store instance (simulates restart)
        store2 = SourceProfileStore(data_dir=Path(tmpdir))
        retrieved = store2.get("persistent_user")

        assert retrieved is not None
        assert retrieved.total_signals == 15


def test_store_cache():
    """Test in-memory cache returns same object without disk read."""
    from src.models.social_message import SourceType
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        profile = SourceProfile(
            author_id="cached_user",
            source_type=SourceType.GROK,
            total_signals=5,
            correct_signals=3,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        store.save(profile)

        # Second get should return from cache
        retrieved1 = store.get("cached_user")
        retrieved2 = store.get("cached_user")

        # Both should be the same object (from cache)
        assert retrieved1 is retrieved2


def test_store_preserves_all_fields():
    """Test all SourceProfile fields are preserved through save/load cycle."""
    from src.models.social_message import SourceType
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        first_seen = datetime(2024, 1, 15, 10, 30, 0)
        last_seen = datetime(2024, 6, 20, 14, 45, 30)

        original = SourceProfile(
            author_id="full_profile_user",
            source_type=SourceType.TWITTER,
            first_seen=first_seen,
            last_seen=last_seen,
            total_signals=25,
            correct_signals=20,
            followers=15000,
            verified=True,
            account_age_days=365,
            category="analyst",
            signals_history=["sig1", "sig2", "sig3"],
        )

        # Save with first store, load with second
        store1 = SourceProfileStore(data_dir=Path(tmpdir))
        store1.save(original)

        store2 = SourceProfileStore(data_dir=Path(tmpdir))
        loaded = store2.get("full_profile_user")

        assert loaded is not None
        assert loaded.author_id == original.author_id
        assert loaded.source_type == original.source_type
        assert loaded.first_seen == original.first_seen
        assert loaded.last_seen == original.last_seen
        assert loaded.total_signals == original.total_signals
        assert loaded.correct_signals == original.correct_signals
        assert loaded.followers == original.followers
        assert loaded.verified == original.verified
        assert loaded.account_age_days == original.account_age_days
        assert loaded.category == original.category
        assert loaded.signals_history == original.signals_history


def test_store_update_existing():
    """Test updating an existing profile overwrites correctly."""
    from src.models.social_message import SourceType
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # Create initial profile
        profile = SourceProfile(
            author_id="update_user",
            source_type=SourceType.GROK,
            total_signals=5,
            correct_signals=3,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store.save(profile)

        # Update the profile
        profile.total_signals = 10
        profile.correct_signals = 8
        store.save(profile)

        # Verify update persisted
        retrieved = store.get("update_user")
        assert retrieved is not None
        assert retrieved.total_signals == 10
        assert retrieved.correct_signals == 8


def test_store_creates_directory():
    """Test store creates data directory if it doesn't exist."""
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "path" / "sources"
        store = SourceProfileStore(data_dir=nested_path)

        # Directory should be created
        assert nested_path.exists()
        assert nested_path.is_dir()
