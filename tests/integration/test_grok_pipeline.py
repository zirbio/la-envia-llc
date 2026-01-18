# tests/integration/test_grok_pipeline.py
"""Integration tests for the Grok-based signal processing pipeline.

Tests the complete flow from Grok collector through credibility management
and outcome tracking feedback loop.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.collectors.grok_collector import GrokCollector
from src.models.social_message import SocialMessage, SourceType
from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
from src.scoring.signal_outcome_tracker import (
    PendingEvaluation,
    SignalOutcomeTracker,
)
from src.scoring.source_profile import SourceProfile
from src.scoring.source_profile_store import SourceProfileStore


class TestGrokSignalToCredibilityUpdate:
    """Test 1: Verify signal from Grok flows through to credibility update."""

    @pytest.fixture
    def test_social_message(self):
        """Create a test SocialMessage from Grok source."""
        return SocialMessage(
            source=SourceType.GROK,
            source_id="grok_test_001",
            author="test_trader",
            content="$NVDA massive call sweep! Very bullish setup!",
            timestamp=datetime.now(timezone.utc),
            url="https://x.com/test_trader/status/grok_test_001",
            like_count=150,
            retweet_count=50,
        )

    @pytest.mark.asyncio
    async def test_grok_signal_to_credibility_update(self, test_social_message):
        """Tests that a signal from Grok collector flows through to credibility update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create real SourceProfileStore (with temp directory)
            store = SourceProfileStore(data_dir=Path(tmpdir))

            # Step 2: Create real DynamicCredibilityManager
            credibility_manager = DynamicCredibilityManager(profile_store=store)

            # Step 3: Create mock Alpaca client for SignalOutcomeTracker
            mock_alpaca = AsyncMock()
            mock_alpaca.get_current_price = AsyncMock(return_value=505.0)  # +1% from 500

            # Step 4: Create real SignalOutcomeTracker
            outcome_tracker = SignalOutcomeTracker(
                credibility_manager=credibility_manager,
                alpaca_client=mock_alpaca,
                evaluation_window_minutes=30,
                success_threshold_percent=1.0,
            )

            # Step 5: Process the signal - record it in credibility manager
            credibility_manager.record_signal(
                author_id=test_social_message.author,
                source_type=test_social_message.source,
                signal_id=test_social_message.source_id,
            )

            # Step 6: Verify credibility manager recorded the signal
            profile = store.get(test_social_message.author)
            assert profile is not None, "Profile should be created for new author"
            assert profile.total_signals == 1, "Signal count should be 1"
            assert test_social_message.source_id in profile.signals_history

            # Step 7: Record signal in outcome tracker for evaluation
            outcome_tracker.record_signal(
                signal_id=test_social_message.source_id,
                author_id=test_social_message.author,
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
            )

            # Step 8: Simulate outcome evaluation by manually setting evaluate_at to past
            assert len(outcome_tracker._pending_evaluations) == 1
            evaluation = outcome_tracker._pending_evaluations[0]
            evaluation.evaluate_at = datetime.now() - timedelta(minutes=5)

            # Step 9: Evaluate pending signals
            await outcome_tracker.evaluate_pending()

            # Step 10: Verify profile accuracy is updated
            updated_profile = store.get(test_social_message.author)
            assert updated_profile is not None
            assert updated_profile.correct_signals == 1, "Correct signal should be recorded"


class TestGrokCollectorToScorer:
    """Test 2: Verify signal scoring with dynamic credibility."""

    @pytest.fixture
    def high_accuracy_profile(self):
        """Create a pre-populated high accuracy author profile."""
        return SourceProfile(
            author_id="expert_trader",
            source_type=SourceType.GROK,
            first_seen=datetime.now() - timedelta(days=30),
            last_seen=datetime.now(),
            total_signals=10,
            correct_signals=8,  # 80% accuracy -> 1.3x multiplier
            followers=50000,
            verified=True,
        )

    @pytest.fixture
    def grok_message_from_expert(self):
        """Create a SocialMessage from the expert trader."""
        return SocialMessage(
            source=SourceType.GROK,
            source_id="expert_signal_001",
            author="expert_trader",
            content="$TSLA breaking out! Strong momentum buy signal!",
            timestamp=datetime.now(timezone.utc),
            url="https://x.com/expert_trader/status/expert_signal_001",
            like_count=1000,
            retweet_count=200,
        )

    @pytest.fixture
    def grok_message_from_unknown(self):
        """Create a SocialMessage from an unknown author."""
        return SocialMessage(
            source=SourceType.GROK,
            source_id="unknown_signal_001",
            author="random_user",
            content="$AAPL to the moon!",
            timestamp=datetime.now(timezone.utc),
        )

    def test_grok_collector_to_scorer_high_accuracy(
        self, high_accuracy_profile, grok_message_from_expert
    ):
        """Tests that high accuracy author gets multiplier > 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store and save high-accuracy profile
            store = SourceProfileStore(data_dir=Path(tmpdir))
            store.save(high_accuracy_profile)

            # Create credibility manager
            credibility_manager = DynamicCredibilityManager(profile_store=store)

            # Get multiplier for high-accuracy author
            multiplier = credibility_manager.get_multiplier(
                author_id=grok_message_from_expert.author,
                source_type=grok_message_from_expert.source,
            )

            # Verify multiplier is > 1.0 (1.3 for >= 75% accuracy)
            assert multiplier > 1.0, "High accuracy author should have multiplier > 1.0"
            assert multiplier == 1.3, "80% accuracy should yield 1.3x multiplier"

    def test_grok_collector_to_scorer_unknown_author(self, grok_message_from_unknown):
        """Tests that unknown author gets default multiplier of 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SourceProfileStore(data_dir=Path(tmpdir))
            credibility_manager = DynamicCredibilityManager(profile_store=store)

            # Get multiplier for unknown author
            multiplier = credibility_manager.get_multiplier(
                author_id=grok_message_from_unknown.author,
                source_type=grok_message_from_unknown.source,
            )

            # Verify multiplier is exactly 1.0 for unknown author
            assert multiplier == 1.0, "Unknown author should have multiplier of 1.0"

    def test_grok_tier1_seed_source(self):
        """Tests that tier1 seed source gets tier1 multiplier without history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SourceProfileStore(data_dir=Path(tmpdir))

            # Create manager with tier1 seed list
            credibility_manager = DynamicCredibilityManager(
                profile_store=store,
                tier1_sources=["unusual_whales", "deitaone", "optionflow"],
                tier1_multiplier=1.3,
            )

            # Get multiplier for tier1 seed source (no history)
            multiplier = credibility_manager.get_multiplier(
                author_id="unusual_whales",
                source_type=SourceType.GROK,
            )

            # Verify tier1 seed gets the tier1 multiplier
            assert multiplier == 1.3, "Tier1 seed source should get 1.3x multiplier"


class TestOutcomeTrackerFeedbackLoop:
    """Test 3: Verify the complete feedback loop."""

    @pytest.mark.asyncio
    async def test_outcome_tracker_feedback_loop_bullish_correct(self):
        """Tests complete feedback loop for a correct bullish signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup components
            store = SourceProfileStore(data_dir=Path(tmpdir))
            credibility_manager = DynamicCredibilityManager(profile_store=store)

            # Mock alpaca to return higher price (bullish correct)
            mock_alpaca = AsyncMock()
            mock_alpaca.get_current_price = AsyncMock(return_value=510.0)  # +2%

            outcome_tracker = SignalOutcomeTracker(
                credibility_manager=credibility_manager,
                alpaca_client=mock_alpaca,
                evaluation_window_minutes=30,
                success_threshold_percent=1.0,
            )

            # Step 1: Record a bullish signal
            author_id = "feedback_trader"
            signal_id = "feedback_signal_001"

            # First create the profile by recording the signal in credibility manager
            credibility_manager.record_signal(author_id, SourceType.GROK, signal_id)

            # Record signal in outcome tracker
            outcome_tracker.record_signal(
                signal_id=signal_id,
                author_id=author_id,
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
            )

            # Verify initial state
            profile_before = store.get(author_id)
            assert profile_before.total_signals == 1
            assert profile_before.correct_signals == 0

            # Step 2: Simulate window passing by modifying evaluate_at
            outcome_tracker._pending_evaluations[0].evaluate_at = datetime.now() - timedelta(
                minutes=5
            )

            # Step 3: Call evaluate_pending
            await outcome_tracker.evaluate_pending()

            # Step 4: Verify profile accuracy increased
            profile_after = store.get(author_id)
            assert profile_after.correct_signals == 1, "Correct signal should be recorded"
            assert profile_after.accuracy > 0, "Accuracy should be > 0"

    @pytest.mark.asyncio
    async def test_outcome_tracker_feedback_loop_bullish_incorrect(self):
        """Tests complete feedback loop for an incorrect bullish signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SourceProfileStore(data_dir=Path(tmpdir))
            credibility_manager = DynamicCredibilityManager(profile_store=store)

            # Mock alpaca to return lower price (bullish incorrect)
            mock_alpaca = AsyncMock()
            mock_alpaca.get_current_price = AsyncMock(return_value=490.0)  # -2%

            outcome_tracker = SignalOutcomeTracker(
                credibility_manager=credibility_manager,
                alpaca_client=mock_alpaca,
                evaluation_window_minutes=30,
                success_threshold_percent=1.0,
            )

            author_id = "bad_prediction_trader"
            signal_id = "bad_signal_001"

            # Record signal
            credibility_manager.record_signal(author_id, SourceType.GROK, signal_id)
            outcome_tracker.record_signal(
                signal_id=signal_id,
                author_id=author_id,
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
            )

            # Simulate window passing
            outcome_tracker._pending_evaluations[0].evaluate_at = datetime.now() - timedelta(
                minutes=5
            )

            # Evaluate
            await outcome_tracker.evaluate_pending()

            # Verify accuracy not increased (signal was incorrect)
            profile_after = store.get(author_id)
            assert profile_after.correct_signals == 0, "Incorrect signal should not increase count"
            assert profile_after.accuracy == 0, "Accuracy should be 0 for all incorrect"

    @pytest.mark.asyncio
    async def test_outcome_tracker_feedback_loop_bearish_correct(self):
        """Tests complete feedback loop for a correct bearish signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SourceProfileStore(data_dir=Path(tmpdir))
            credibility_manager = DynamicCredibilityManager(profile_store=store)

            # Mock alpaca to return lower price (bearish correct)
            mock_alpaca = AsyncMock()
            mock_alpaca.get_current_price = AsyncMock(return_value=490.0)  # -2%

            outcome_tracker = SignalOutcomeTracker(
                credibility_manager=credibility_manager,
                alpaca_client=mock_alpaca,
                evaluation_window_minutes=30,
                success_threshold_percent=1.0,
            )

            author_id = "bear_trader"
            signal_id = "bear_signal_001"

            # Record bearish signal
            credibility_manager.record_signal(author_id, SourceType.GROK, signal_id)
            outcome_tracker.record_signal(
                signal_id=signal_id,
                author_id=author_id,
                symbol="NVDA",
                direction="bearish",
                entry_price=500.0,
            )

            # Simulate window passing
            outcome_tracker._pending_evaluations[0].evaluate_at = datetime.now() - timedelta(
                minutes=5
            )

            # Evaluate
            await outcome_tracker.evaluate_pending()

            # Verify bearish correct was recorded
            profile_after = store.get(author_id)
            assert profile_after.correct_signals == 1, "Correct bearish signal should be recorded"

    @pytest.mark.asyncio
    async def test_feedback_loop_updates_multiplier_over_time(self):
        """Tests that multiplier changes as more signals are evaluated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SourceProfileStore(data_dir=Path(tmpdir))
            credibility_manager = DynamicCredibilityManager(
                profile_store=store, min_signals_for_ranking=5
            )

            mock_alpaca = AsyncMock()
            mock_alpaca.get_current_price = AsyncMock(return_value=510.0)  # Always +2%

            outcome_tracker = SignalOutcomeTracker(
                credibility_manager=credibility_manager,
                alpaca_client=mock_alpaca,
                evaluation_window_minutes=30,
                success_threshold_percent=1.0,
            )

            author_id = "consistent_trader"

            # Record and evaluate 5 correct bullish signals
            for i in range(5):
                signal_id = f"signal_{i:03d}"
                credibility_manager.record_signal(author_id, SourceType.GROK, signal_id)
                outcome_tracker.record_signal(
                    signal_id=signal_id,
                    author_id=author_id,
                    symbol="NVDA",
                    direction="bullish",
                    entry_price=500.0,
                )
                # Immediately make it ready for evaluation
                outcome_tracker._pending_evaluations[-1].evaluate_at = datetime.now() - timedelta(
                    minutes=5
                )
                await outcome_tracker.evaluate_pending()

            # After 5 correct signals, multiplier should be 1.3 (>= 75% accuracy)
            profile = store.get(author_id)
            assert profile.total_signals == 5
            assert profile.correct_signals == 5
            assert profile.accuracy == 1.0

            multiplier = credibility_manager.get_multiplier(author_id, SourceType.GROK)
            assert multiplier == 1.3, "100% accuracy with 5+ signals should yield 1.3x"


class TestGrokCollectorIntegration:
    """Additional tests for GrokCollector integration."""

    @pytest.mark.asyncio
    async def test_mock_grok_collector_yields_messages(self):
        """Tests that a mock GrokCollector can yield SocialMessages."""
        # Create a mock collector that yields a test message
        test_message = SocialMessage(
            source=SourceType.GROK,
            source_id="mock_001",
            author="mock_trader",
            content="$AAPL looking strong! Buy signal!",
            timestamp=datetime.now(timezone.utc),
        )

        # Create mock async iterator
        async def mock_stream():
            yield test_message

        # Simulate collector behavior
        mock_collector = MagicMock(spec=GrokCollector)
        mock_collector.stream = mock_stream

        # Process messages from collector
        messages = []
        async for msg in mock_collector.stream():
            messages.append(msg)
            break  # Just get one message for testing

        assert len(messages) == 1
        assert messages[0].source == SourceType.GROK
        assert messages[0].author == "mock_trader"

    @pytest.mark.asyncio
    async def test_end_to_end_grok_signal_flow(self):
        """Tests complete flow: Grok message -> credibility -> outcome tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup all components
            store = SourceProfileStore(data_dir=Path(tmpdir))
            credibility_manager = DynamicCredibilityManager(
                profile_store=store,
                tier1_sources=["unusual_whales"],
                tier1_multiplier=1.3,
            )

            mock_alpaca = AsyncMock()
            mock_alpaca.get_current_price = AsyncMock(return_value=152.0)  # +1.33%

            outcome_tracker = SignalOutcomeTracker(
                credibility_manager=credibility_manager,
                alpaca_client=mock_alpaca,
                evaluation_window_minutes=30,
                success_threshold_percent=1.0,
            )

            # Simulate receiving a Grok message
            grok_message = SocialMessage(
                source=SourceType.GROK,
                source_id="e2e_001",
                author="unusual_whales",
                content="$AAPL massive call sweep detected! Very bullish!",
                timestamp=datetime.now(timezone.utc),
            )

            # Step 1: Check initial multiplier (tier1 seed)
            initial_multiplier = credibility_manager.get_multiplier(
                grok_message.author, grok_message.source
            )
            assert initial_multiplier == 1.3, "Tier1 seed should have 1.3x multiplier"

            # Step 2: Record the signal
            credibility_manager.record_signal(
                grok_message.author, grok_message.source, grok_message.source_id
            )

            # Step 3: Record in outcome tracker (simulating trade execution)
            outcome_tracker.record_signal(
                signal_id=grok_message.source_id,
                author_id=grok_message.author,
                symbol="AAPL",
                direction="bullish",
                entry_price=150.0,
            )

            # Step 4: Simulate evaluation window passing
            outcome_tracker._pending_evaluations[0].evaluate_at = datetime.now() - timedelta(
                minutes=5
            )

            # Step 5: Evaluate outcome
            await outcome_tracker.evaluate_pending()

            # Step 6: Verify profile was updated
            profile = store.get(grok_message.author)
            assert profile.total_signals == 1
            assert profile.correct_signals == 1
            assert profile.accuracy == 1.0

            # Note: Multiplier won't change yet because min_signals_for_ranking defaults to 5
            # But profile is correctly tracking the outcome
