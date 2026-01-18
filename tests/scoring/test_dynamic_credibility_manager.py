# tests/scoring/test_dynamic_credibility_manager.py
"""Tests for DynamicCredibilityManager."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest


def test_get_multiplier_unknown_author():
    """Test unknown author gets default multiplier of 1.0."""
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        multiplier = manager.get_multiplier("unknown_author", SourceType.GROK)
        assert multiplier == 1.0  # Default for unknown


def test_get_multiplier_tier1_seed():
    """Test tier1 seed source gets tier1 multiplier without history."""
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(
            profile_store=store,
            tier1_sources=["unusual_whales"],
            tier1_multiplier=1.3,
        )

        # No history but in tier1 seeds
        multiplier = manager.get_multiplier("unusual_whales", SourceType.GROK)
        assert multiplier == 1.3


def test_get_multiplier_dynamic_from_history():
    """Test author with sufficient history gets dynamic multiplier."""
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # Create profile with history
        profile = SourceProfile(
            author_id="experienced_trader",
            source_type=SourceType.GROK,
            total_signals=10,
            correct_signals=8,  # 80% accuracy
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store.save(profile)

        manager = DynamicCredibilityManager(profile_store=store)
        multiplier = manager.get_multiplier("experienced_trader", SourceType.GROK)
        assert multiplier == 1.3  # >= 75% accuracy


def test_record_signal():
    """Test recording a signal creates/updates profile."""
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        manager.record_signal("new_author", SourceType.GROK, "signal_001")

        profile = store.get("new_author")
        assert profile is not None
        assert profile.total_signals == 1
        assert "signal_001" in profile.signals_history


def test_record_outcome():
    """Test recording outcome updates correct_signals count."""
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # Create profile with some signals
        profile = SourceProfile(
            author_id="trader",
            source_type=SourceType.GROK,
            total_signals=5,
            correct_signals=3,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store.save(profile)

        manager = DynamicCredibilityManager(profile_store=store)

        # Record correct outcome
        manager.record_outcome("trader", was_correct=True)

        updated = store.get("trader")
        assert updated.correct_signals == 4
        assert updated.accuracy == 0.8  # 4/5
