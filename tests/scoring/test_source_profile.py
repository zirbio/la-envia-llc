# tests/scoring/test_source_profile.py
"""Tests for SourceProfile model."""
from datetime import datetime
import pytest


def test_source_profile_creation():
    """Test basic SourceProfile creation with default values."""
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="unusual_whales",
        source_type=SourceType.GROK,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    assert profile.author_id == "unusual_whales"
    assert profile.source_type == SourceType.GROK
    assert profile.total_signals == 0
    assert profile.correct_signals == 0
    assert profile.accuracy == 0.0


def test_credibility_multiplier_insufficient_data():
    """Test credibility multiplier returns 1.0 with insufficient data."""
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="new_trader",
        source_type=SourceType.GROK,
        total_signals=3,
        correct_signals=3,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    # Less than 5 signals = default 1.0
    assert profile.credibility_multiplier == 1.0


def test_credibility_multiplier_high_accuracy():
    """Test credibility multiplier for high accuracy (>= 75%)."""
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="expert_trader",
        source_type=SourceType.GROK,
        total_signals=20,
        correct_signals=16,  # 80% accuracy
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    assert profile.accuracy == 0.8
    assert profile.credibility_multiplier == 1.3  # >= 75%


def test_credibility_multiplier_low_accuracy():
    """Test credibility multiplier for low accuracy (< 40%)."""
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="bad_trader",
        source_type=SourceType.GROK,
        total_signals=10,
        correct_signals=3,  # 30% accuracy
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    assert profile.accuracy == 0.3
    assert profile.credibility_multiplier == 0.5  # < 40%
