# Phase 4: Scoring System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a scoring system (CAPA 4) that combines sentiment analysis and technical validation to produce trade recommendations with entry, stop loss, and take profit levels.

**Architecture:** SignalScorer receives ValidatedSignal from Phase 3, applies dynamic weighting based on market conditions, calculates confluence and credibility bonuses, and outputs TradeRecommendation with full trade parameters.

**Tech Stack:** Python dataclasses, Pydantic (settings), pytest (TDD)

---

## Task 1: Data Models

**Files:**
- Create: `src/scoring/__init__.py`
- Create: `src/scoring/models.py`
- Test: `tests/scoring/__init__.py`
- Test: `tests/scoring/test_models.py`

**Requirements:**
Create data models for the scoring system:

1. `Direction` enum: LONG, SHORT, NEUTRAL
2. `ScoreTier` enum: STRONG, MODERATE, WEAK, NO_TRADE
3. `ScoreComponents` dataclass:
   - sentiment_score: float (0-100)
   - technical_score: float (0-100)
   - sentiment_weight: float (0-1)
   - technical_weight: float (0-1)
   - confluence_bonus: float (0-0.2)
   - credibility_multiplier: float (0.8-1.2)
   - time_factor: float (0.5-1.0)

4. `TradeRecommendation` dataclass:
   - symbol: str
   - direction: Direction
   - score: float (0-100)
   - tier: ScoreTier
   - position_size_percent: float (0-100)
   - entry_price: float
   - stop_loss: float
   - take_profit: float
   - risk_reward_ratio: float
   - components: ScoreComponents
   - reasoning: str
   - timestamp: datetime

**Test Cases:**
```python
def test_direction_enum_values():
    assert Direction.LONG.value == "long"
    assert Direction.SHORT.value == "short"
    assert Direction.NEUTRAL.value == "neutral"

def test_score_tier_enum_values():
    assert ScoreTier.STRONG.value == "strong"
    assert ScoreTier.MODERATE.value == "moderate"

def test_trade_recommendation_creation():
    # Create with all fields, verify structure

def test_score_tier_from_score():
    # Helper method to get tier from numeric score
```

**TDD Steps:**
1. Write failing tests
2. Implement models
3. Verify tests pass
4. Commit: "feat(scoring): add data models for scoring system"

---

## Task 2: Source Credibility Manager

**Files:**
- Create: `src/scoring/source_credibility.py`
- Test: `tests/scoring/test_source_credibility.py`

**Requirements:**
```python
class SourceCredibilityManager:
    """Maps authors to credibility tiers and multipliers."""

    def __init__(
        self,
        tier1_sources: list[str] = None,  # Premium sources
        tier1_multiplier: float = 1.2,
        tier2_multiplier: float = 1.0,
        tier3_multiplier: float = 0.8,
    ):
        pass

    def get_multiplier(self, author: str, source: SourceType) -> float:
        """Get credibility multiplier for an author."""

    def get_tier(self, author: str) -> int:
        """Get tier (1, 2, or 3) for an author."""

    def add_tier1_source(self, author: str) -> None:
        """Dynamically add a trusted source."""
```

**Default Tier 1 Sources:**
- unusual_whales
- optionsflow
- wallstreetbets (moderators only)
- thestreet
- zaborskierik

**Test Cases:**
1. `test_tier1_source_returns_high_multiplier`
2. `test_unknown_source_returns_tier3_multiplier`
3. `test_add_tier1_source_dynamically`
4. `test_source_type_affects_credibility` (Twitter vs Reddit)

**Commit:** "feat(scoring): add source credibility manager"

---

## Task 3: Time Factors Calculator

**Files:**
- Create: `src/scoring/time_factors.py`
- Test: `tests/scoring/test_time_factors.py`

**Requirements:**
```python
from datetime import datetime, time
from zoneinfo import ZoneInfo

class TimeFactorCalculator:
    """Calculates time-based penalties for signals."""

    def __init__(
        self,
        timezone: str = "America/New_York",
        premarket_factor: float = 0.9,
        market_hours_factor: float = 1.0,
        afterhours_factor: float = 0.8,
        earnings_factor: float = 0.7,
        earnings_proximity_days: int = 3,
    ):
        pass

    def calculate_factor(
        self,
        timestamp: datetime,
        symbol: str,
        earnings_dates: dict[str, datetime] = None,
    ) -> tuple[float, list[str]]:
        """
        Calculate time factor and return reasons.
        Returns: (factor, list of reasons applied)
        """

    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours."""

    def is_premarket(self, timestamp: datetime) -> bool:
        """Check if timestamp is during pre-market (4am-9:30am ET)."""
```

**Market Hours (ET):**
- Pre-market: 4:00 AM - 9:30 AM → 0.9x
- Regular: 9:30 AM - 4:00 PM → 1.0x
- After-hours: 4:00 PM - 8:00 PM → 0.8x
- Closed: 8:00 PM - 4:00 AM → 0.5x

**Test Cases:**
1. `test_market_hours_returns_full_factor`
2. `test_premarket_returns_reduced_factor`
3. `test_afterhours_returns_reduced_factor`
4. `test_earnings_proximity_applies_penalty`
5. `test_weekend_returns_lowest_factor`

**Commit:** "feat(scoring): add time factors calculator"

---

## Task 4: Confluence Detector

**Files:**
- Create: `src/scoring/confluence_detector.py`
- Test: `tests/scoring/test_confluence_detector.py`

**Requirements:**
```python
from collections import defaultdict
from datetime import datetime, timedelta

class ConfluenceDetector:
    """Tracks multiple signals on same ticker within time window."""

    def __init__(
        self,
        window_minutes: int = 15,
        bonus_2_signals: float = 0.10,
        bonus_3_signals: float = 0.20,
    ):
        self._signals: dict[str, list[datetime]] = defaultdict(list)

    def record_signal(self, symbol: str, timestamp: datetime) -> None:
        """Record a new signal for a symbol."""

    def get_confluence_count(self, symbol: str, timestamp: datetime) -> int:
        """Get number of signals for symbol within window."""

    def get_bonus(self, symbol: str, timestamp: datetime) -> float:
        """Get confluence bonus (0.0, 0.10, or 0.20)."""

    def cleanup_old_signals(self) -> None:
        """Remove signals outside the window."""
```

**Test Cases:**
1. `test_single_signal_no_bonus`
2. `test_two_signals_within_window_returns_bonus`
3. `test_three_signals_returns_higher_bonus`
4. `test_signals_outside_window_not_counted`
5. `test_cleanup_removes_old_signals`

**Commit:** "feat(scoring): add confluence detector"

---

## Task 5: Dynamic Weight Calculator

**Files:**
- Create: `src/scoring/dynamic_weight_calculator.py`
- Test: `tests/scoring/test_dynamic_weight_calculator.py`

**Requirements:**
```python
class DynamicWeightCalculator:
    """Calculates sentiment/technical weights based on market conditions."""

    def __init__(
        self,
        base_sentiment_weight: float = 0.5,
        base_technical_weight: float = 0.5,
        strong_trend_adx: float = 30.0,
        weak_trend_adx: float = 20.0,
        strong_trend_technical_weight: float = 0.6,
        weak_trend_sentiment_weight: float = 0.6,
    ):
        pass

    def calculate_weights(
        self,
        adx: float,
        volatility_percentile: float = 50.0,  # Optional: 0-100
    ) -> tuple[float, float]:
        """
        Calculate sentiment and technical weights.
        Returns: (sentiment_weight, technical_weight) that sum to 1.0
        """
```

**Weight Rules:**
| Condition | Sentiment | Technical |
|-----------|-----------|-----------|
| ADX > 30 (strong trend) | 0.4 | 0.6 |
| ADX 20-30 (normal) | 0.5 | 0.5 |
| ADX < 20 (weak trend) | 0.6 | 0.4 |
| High volatility override | 0.35 | 0.65 |

**Test Cases:**
1. `test_strong_trend_favors_technical`
2. `test_weak_trend_favors_sentiment`
3. `test_normal_trend_balanced_weights`
4. `test_weights_always_sum_to_one`
5. `test_high_volatility_overrides_adx`

**Commit:** "feat(scoring): add dynamic weight calculator"

---

## Task 6: Recommendation Builder

**Files:**
- Create: `src/scoring/recommendation_builder.py`
- Test: `tests/scoring/test_recommendation_builder.py`

**Requirements:**
```python
class RecommendationBuilder:
    """Builds trade recommendations with entry, stop, and target."""

    def __init__(
        self,
        default_stop_loss_percent: float = 2.0,
        default_risk_reward_ratio: float = 2.0,
        tier_strong_threshold: int = 80,
        tier_moderate_threshold: int = 60,
        tier_weak_threshold: int = 40,
        position_size_strong: float = 100.0,
        position_size_moderate: float = 50.0,
        position_size_weak: float = 25.0,
    ):
        pass

    def get_tier(self, score: float) -> ScoreTier:
        """Get score tier from numeric score."""

    def get_position_size(self, tier: ScoreTier) -> float:
        """Get position size percentage for tier."""

    def calculate_levels(
        self,
        entry_price: float,
        direction: Direction,
        stop_loss_percent: float = None,
        risk_reward_ratio: float = None,
    ) -> tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit.
        Returns: (entry, stop_loss, take_profit)
        """

    def build(
        self,
        symbol: str,
        direction: Direction,
        score: float,
        current_price: float,
        components: ScoreComponents,
        reasoning: str,
    ) -> TradeRecommendation:
        """Build complete trade recommendation."""
```

**Test Cases:**
1. `test_tier_from_score_strong`
2. `test_tier_from_score_moderate`
3. `test_tier_from_score_weak`
4. `test_tier_from_score_no_trade`
5. `test_calculate_levels_long`
6. `test_calculate_levels_short`
7. `test_build_returns_complete_recommendation`

**Commit:** "feat(scoring): add recommendation builder"

---

## Task 7: Signal Scorer (Orchestrator)

**Files:**
- Create: `src/scoring/signal_scorer.py`
- Test: `tests/scoring/test_signal_scorer.py`

**Requirements:**
```python
class SignalScorer:
    """Orchestrates the scoring pipeline."""

    def __init__(
        self,
        credibility_manager: SourceCredibilityManager,
        time_calculator: TimeFactorCalculator,
        confluence_detector: ConfluenceDetector,
        weight_calculator: DynamicWeightCalculator,
        recommendation_builder: RecommendationBuilder,
    ):
        pass

    def score(
        self,
        validated_signal: ValidatedSignal,
        current_price: float,
        earnings_dates: dict[str, datetime] = None,
    ) -> TradeRecommendation:
        """
        Score a validated signal and produce trade recommendation.

        Pipeline:
        1. Extract sentiment score from Phase 2
        2. Extract technical score from Phase 3
        3. Calculate dynamic weights based on ADX
        4. Apply credibility multiplier
        5. Apply time factor
        6. Apply confluence bonus
        7. Calculate final score
        8. Build trade recommendation
        """

    def _calculate_sentiment_score(self, validated_signal: ValidatedSignal) -> float:
        """Convert sentiment analysis to 0-100 score."""

    def _calculate_technical_score(self, validated_signal: ValidatedSignal) -> float:
        """Convert technical validation to 0-100 score."""

    def _determine_direction(self, validated_signal: ValidatedSignal) -> Direction:
        """Determine trade direction from sentiment."""
```

**Scoring Formula:**
```python
base_score = (sentiment_score * sentiment_weight) + (technical_score * technical_weight)
final_score = base_score * credibility_multiplier * time_factor * (1 + confluence_bonus)
final_score = min(100, max(0, final_score))  # Clamp to 0-100
```

**Test Cases:**
1. `test_score_returns_trade_recommendation`
2. `test_score_applies_dynamic_weights`
3. `test_score_applies_credibility_multiplier`
4. `test_score_applies_time_factor`
5. `test_score_applies_confluence_bonus`
6. `test_veto_signal_returns_no_trade`
7. `test_neutral_sentiment_returns_neutral_direction`

**Commit:** "feat(scoring): add SignalScorer orchestrator"

---

## Task 8: Scoring Settings

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Test: `tests/config/test_scoring_settings.py`

**Requirements:**
Add ScoringSettings to config:

```python
class ScoringSettings(BaseModel):
    """Settings for scoring system."""

    enabled: bool = True

    # Score thresholds
    tier_strong_threshold: int = Field(default=80, ge=0, le=100)
    tier_moderate_threshold: int = Field(default=60, ge=0, le=100)
    tier_weak_threshold: int = Field(default=40, ge=0, le=100)

    # Position sizing
    position_size_strong: float = Field(default=100.0, ge=0, le=100)
    position_size_moderate: float = Field(default=50.0, ge=0, le=100)
    position_size_weak: float = Field(default=25.0, ge=0, le=100)

    # Trade parameters
    default_stop_loss_percent: float = Field(default=2.0, ge=0.1, le=10.0)
    default_risk_reward_ratio: float = Field(default=2.0, ge=1.0, le=5.0)

    # Dynamic weights
    base_sentiment_weight: float = Field(default=0.5, ge=0, le=1)
    base_technical_weight: float = Field(default=0.5, ge=0, le=1)
    strong_trend_adx: float = Field(default=30.0, ge=20, le=50)
    weak_trend_adx: float = Field(default=20.0, ge=10, le=30)

    # Confluence
    confluence_window_minutes: int = Field(default=15, ge=1, le=60)
    confluence_bonus_2_signals: float = Field(default=0.10, ge=0, le=0.5)
    confluence_bonus_3_signals: float = Field(default=0.20, ge=0, le=0.5)

    # Time factors
    premarket_factor: float = Field(default=0.9, ge=0.5, le=1.0)
    afterhours_factor: float = Field(default=0.8, ge=0.5, le=1.0)
    earnings_proximity_days: int = Field(default=3, ge=1, le=14)
    earnings_factor: float = Field(default=0.7, ge=0.5, le=1.0)

    # Source credibility
    credibility_tier1_multiplier: float = Field(default=1.2, ge=1.0, le=1.5)
    credibility_tier2_multiplier: float = Field(default=1.0, ge=0.8, le=1.2)
    credibility_tier3_multiplier: float = Field(default=0.8, ge=0.5, le=1.0)
    tier1_sources: list[str] = Field(default_factory=lambda: [
        "unusual_whales",
        "optionsflow",
    ])
```

**Test Cases:**
1. `test_settings_has_scoring_config`
2. `test_default_values`
3. `test_threshold_validation`
4. `test_position_size_validation`

**Commit:** "feat(config): add scoring settings"

---

## Task 9: Update Module Exports

**Files:**
- Modify: `src/scoring/__init__.py`

**Requirements:**
```python
# src/scoring/__init__.py
"""Scoring module for CAPA 4."""

from .models import Direction, ScoreTier, ScoreComponents, TradeRecommendation
from .source_credibility import SourceCredibilityManager
from .time_factors import TimeFactorCalculator
from .confluence_detector import ConfluenceDetector
from .dynamic_weight_calculator import DynamicWeightCalculator
from .recommendation_builder import RecommendationBuilder
from .signal_scorer import SignalScorer

__all__ = [
    "Direction",
    "ScoreTier",
    "ScoreComponents",
    "TradeRecommendation",
    "SourceCredibilityManager",
    "TimeFactorCalculator",
    "ConfluenceDetector",
    "DynamicWeightCalculator",
    "RecommendationBuilder",
    "SignalScorer",
]
```

**Commit:** "chore(scoring): update module exports"

---

## Task 10: Integration Tests

**Files:**
- Create: `tests/integration/test_scoring_pipeline.py`

**Requirements:**
Test the full Phase 3 → Phase 4 pipeline:

```python
class TestScoringPipelineIntegration:
    def test_full_pipeline_strong_signal(self):
        """ValidatedSignal with high sentiment + good technicals → STRONG recommendation."""

    def test_full_pipeline_moderate_signal(self):
        """ValidatedSignal with moderate confidence → MODERATE recommendation."""

    def test_full_pipeline_veto_signal(self):
        """ValidatedSignal with VETO status → NO_TRADE recommendation."""

    def test_confluence_boost(self):
        """Multiple signals on same ticker increase score."""

    def test_credibility_affects_score(self):
        """Tier 1 source gets higher score than Tier 3."""

    def test_time_factor_reduces_afterhours(self):
        """After-hours signal gets reduced score."""

    def test_dynamic_weights_strong_trend(self):
        """High ADX weights technical more heavily."""

    def test_recommendation_has_valid_levels(self):
        """Entry, stop loss, take profit are correctly calculated."""
```

**Commit:** "test: add integration tests for scoring pipeline"

---

## Task 11: Final Test Suite Run

**Steps:**
1. Run: `uv run pytest --cov=src --cov-report=term-missing -v`
2. Verify all tests pass
3. Verify >90% coverage on new scoring module
4. Commit if any cleanup needed

---

## Summary

Phase 4 adds the scoring system (CAPA 4) with:

1. **Data Models** - Direction, ScoreTier, TradeRecommendation
2. **Source Credibility** - Author trust levels
3. **Time Factors** - Market hours penalties
4. **Confluence Detector** - Multiple signal bonuses
5. **Dynamic Weight Calculator** - ADX-based weight adjustment
6. **Recommendation Builder** - Entry/stop/target calculation
7. **Signal Scorer** - Main orchestrator
8. **Settings** - Full configuration support
9. **Integration Tests** - Full pipeline testing

**Output:**
```python
TradeRecommendation(
    symbol="NVDA",
    direction=Direction.LONG,
    score=85.5,
    tier=ScoreTier.STRONG,
    position_size_percent=100.0,
    entry_price=142.50,
    stop_loss=139.65,     # -2%
    take_profit=148.20,   # +4% (2:1 R:R)
    risk_reward_ratio=2.0,
    reasoning="Strong bullish sentiment (92%) confirmed by favorable technicals (ADX 28). Tier 1 source (unusual_whales). 2 confluence signals in 15min window.",
)
```
