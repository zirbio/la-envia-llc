# tests/scoring/test_source_credibility.py
"""Tests for source credibility manager."""
import pytest

from src.models.social_message import SourceType
from src.scoring.source_credibility import SourceCredibilityManager


class TestSourceCredibilityManager:
    """Tests for SourceCredibilityManager."""

    def test_tier1_source_returns_high_multiplier(self):
        """Test that unusual_whales (tier1) returns 1.2 multiplier."""
        manager = SourceCredibilityManager()

        multiplier = manager.get_multiplier("unusual_whales")

        assert multiplier == 1.2

    def test_unknown_source_returns_tier3_multiplier(self):
        """Test that random_user (unknown) returns 0.8 multiplier."""
        manager = SourceCredibilityManager()

        multiplier = manager.get_multiplier("random_user")

        assert multiplier == 0.8

    def test_add_tier1_source_dynamically(self):
        """Test adding a source dynamically promotes it to tier1."""
        manager = SourceCredibilityManager()

        # Initially unknown, should be tier 3
        assert manager.get_multiplier("new_trusted_source") == 0.8
        assert manager.get_tier("new_trusted_source") == 3

        # Add to tier 1
        manager.add_tier1_source("new_trusted_source")

        # Now should be tier 1
        assert manager.get_multiplier("new_trusted_source") == 1.2
        assert manager.get_tier("new_trusted_source") == 1

    def test_source_type_affects_credibility(self):
        """Test that source type is passed to get_multiplier (optional param)."""
        manager = SourceCredibilityManager()

        # Verify source_type parameter is accepted
        multiplier = manager.get_multiplier(
            "unusual_whales", source=SourceType.TWITTER
        )
        assert multiplier == 1.2

        # Unknown source with source type
        multiplier = manager.get_multiplier(
            "random_user", source=SourceType.REDDIT
        )
        assert multiplier == 0.8

    def test_get_tier_returns_correct_tier(self):
        """Test get_tier returns 1, 2, or 3 correctly."""
        manager = SourceCredibilityManager()

        # Default tier1 sources should return tier 1
        assert manager.get_tier("unusual_whales") == 1
        assert manager.get_tier("optionsflow") == 1
        assert manager.get_tier("wallstreetbets") == 1
        assert manager.get_tier("thestreet") == 1
        assert manager.get_tier("zaborskierik") == 1

        # Unknown sources should return tier 3
        assert manager.get_tier("some_random_user") == 3
        assert manager.get_tier("another_unknown") == 3

    def test_custom_multipliers(self):
        """Test with custom tier1 and tier3 multipliers."""
        manager = SourceCredibilityManager(
            tier1_multiplier=1.5,
            tier3_multiplier=0.5,
        )

        # Tier1 source with custom multiplier
        assert manager.get_multiplier("unusual_whales") == 1.5

        # Unknown source with custom multiplier
        assert manager.get_multiplier("random_user") == 0.5

    def test_all_default_tier1_sources(self):
        """Test all default tier1 sources return correct multiplier."""
        manager = SourceCredibilityManager()

        default_tier1 = [
            "unusual_whales",
            "optionsflow",
            "wallstreetbets",
            "thestreet",
            "zaborskierik",
        ]

        for source in default_tier1:
            assert manager.get_multiplier(source) == 1.2, f"{source} should be tier1"
            assert manager.get_tier(source) == 1, f"{source} should be tier 1"

    def test_custom_tier1_sources_override_defaults(self):
        """Test that providing custom tier1_sources works correctly."""
        custom_sources = ["my_source", "another_source"]
        manager = SourceCredibilityManager(tier1_sources=custom_sources)

        # Custom sources should be tier1
        assert manager.get_multiplier("my_source") == 1.2
        assert manager.get_multiplier("another_source") == 1.2

        # Default sources should NOT be tier1 anymore
        assert manager.get_multiplier("unusual_whales") == 0.8

    def test_tier2_multiplier_stored(self):
        """Test that tier2_multiplier is stored for future use."""
        manager = SourceCredibilityManager(tier2_multiplier=1.1)

        # Tier2 is for future use, but multiplier should be stored
        # We verify the manager accepts the parameter without error
        assert manager.get_multiplier("unusual_whales") == 1.2
