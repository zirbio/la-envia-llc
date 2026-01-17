"""Tests for orchestrator data models."""

from datetime import datetime

import pytest

from orchestrator.models import AggregatedSentiment, OrchestratorState, ProcessResult


class TestOrchestratorState:
    """Tests for OrchestratorState enum."""

    def test_has_stopped_state(self) -> None:
        assert OrchestratorState.STOPPED.value == "stopped"

    def test_has_running_state(self) -> None:
        assert OrchestratorState.RUNNING.value == "running"

    def test_has_stopping_state(self) -> None:
        assert OrchestratorState.STOPPING.value == "stopping"


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_create_executed_result(self) -> None:
        result = ProcessResult(status="executed", symbol="AAPL")
        assert result.status == "executed"
        assert result.symbol == "AAPL"
        assert result.error is None

    def test_create_error_result(self) -> None:
        result = ProcessResult(status="error", error="Scoring failed")
        assert result.status == "error"
        assert result.error == "Scoring failed"

    def test_create_vetoed_result(self) -> None:
        result = ProcessResult(status="vetoed", symbol="TSLA")
        assert result.status == "vetoed"

    def test_timestamp_auto_set(self) -> None:
        result = ProcessResult(status="buffered")
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestAggregatedSentiment:
    """Tests for AggregatedSentiment dataclass."""

    def test_create_aggregated_sentiment(self) -> None:
        agg = AggregatedSentiment(
            symbol="AAPL",
            bullish_count=5,
            bearish_count=2,
            neutral_count=1,
            total_count=8,
            consensus_label="bullish",
            consensus_strength=0.625,
            avg_confidence=0.75,
        )
        assert agg.symbol == "AAPL"
        assert agg.consensus_label == "bullish"
        assert agg.consensus_strength == 0.625

    def test_consensus_strength_calculation(self) -> None:
        agg = AggregatedSentiment(
            symbol="TSLA",
            bullish_count=8,
            bearish_count=2,
            neutral_count=0,
            total_count=10,
            consensus_label="bullish",
            consensus_strength=0.8,
            avg_confidence=0.85,
        )
        assert agg.consensus_strength == 0.8
