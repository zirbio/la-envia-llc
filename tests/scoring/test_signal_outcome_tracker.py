# tests/scoring/test_signal_outcome_tracker.py
"""Tests for SignalOutcomeTracker."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_record_signal():
    """Test recording a signal for future evaluation."""
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)
        mock_alpaca = MagicMock()

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
        )

        tracker.record_signal(
            signal_id="sig_001",
            author_id="trader123",
            symbol="NVDA",
            direction="bullish",
            entry_price=500.0,
        )

        assert len(tracker._pending_evaluations) == 1
        eval = tracker._pending_evaluations[0]
        assert eval.author_id == "trader123"
        assert eval.symbol == "NVDA"
        assert eval.direction == "bullish"


@pytest.mark.asyncio
async def test_evaluate_correct_bullish():
    """Test evaluating a correct bullish signal."""
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=510.0)  # +2%

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
            success_threshold_percent=1.0,
        )

        # Record signal first to create profile
        manager.record_signal("trader123", SourceType.GROK, "sig_001")

        # Add pending evaluation that's ready
        tracker._pending_evaluations.append(
            PendingEvaluation(
                signal_id="sig_001",
                author_id="trader123",
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
                entry_time=datetime.now() - timedelta(minutes=35),
                evaluate_at=datetime.now() - timedelta(minutes=5),
            )
        )

        await tracker.evaluate_pending()

        # Check outcome was recorded
        profile = store.get("trader123")
        assert profile.correct_signals == 1


@pytest.mark.asyncio
async def test_evaluate_incorrect_bullish():
    """Test evaluating an incorrect bullish signal."""
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=490.0)  # -2%

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
            success_threshold_percent=1.0,
        )

        # Record signal first
        manager.record_signal("bad_trader", SourceType.GROK, "sig_002")

        # Add pending evaluation
        tracker._pending_evaluations.append(
            PendingEvaluation(
                signal_id="sig_002",
                author_id="bad_trader",
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
                entry_time=datetime.now() - timedelta(minutes=35),
                evaluate_at=datetime.now() - timedelta(minutes=5),
            )
        )

        await tracker.evaluate_pending()

        # Check outcome was recorded as incorrect
        profile = store.get("bad_trader")
        assert profile.correct_signals == 0  # Not incremented


@pytest.mark.asyncio
async def test_evaluate_correct_bearish():
    """Test evaluating a correct bearish signal (price went down)."""
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=490.0)  # -2%

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
            success_threshold_percent=1.0,
        )

        # Record signal first to create profile
        manager.record_signal("bear_trader", SourceType.GROK, "sig_003")

        # Add pending evaluation for bearish signal
        tracker._pending_evaluations.append(
            PendingEvaluation(
                signal_id="sig_003",
                author_id="bear_trader",
                symbol="NVDA",
                direction="bearish",
                entry_price=500.0,
                entry_time=datetime.now() - timedelta(minutes=35),
                evaluate_at=datetime.now() - timedelta(minutes=5),
            )
        )

        await tracker.evaluate_pending()

        # Check outcome was recorded as correct (price went down as predicted)
        profile = store.get("bear_trader")
        assert profile.correct_signals == 1


@pytest.mark.asyncio
async def test_evaluate_skips_not_ready():
    """Test that signals not ready for evaluation are skipped."""
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=510.0)

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
        )

        # Record signal first
        manager.record_signal("patient_trader", SourceType.GROK, "sig_004")

        # Add pending evaluation that's NOT ready yet (evaluate_at in future)
        tracker._pending_evaluations.append(
            PendingEvaluation(
                signal_id="sig_004",
                author_id="patient_trader",
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
                entry_time=datetime.now(),
                evaluate_at=datetime.now() + timedelta(minutes=25),  # Future
            )
        )

        await tracker.evaluate_pending()

        # Signal should still be pending (not evaluated)
        assert len(tracker._pending_evaluations) == 1
        # No outcome recorded
        profile = store.get("patient_trader")
        assert profile.correct_signals == 0
