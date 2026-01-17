# Phase 2: Sentiment Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CAPA 2 (Processing & Analysis) with FinTwitBERT sentiment analysis and Claude API deep analysis for social media messages.

**Architecture:** Two-stage analysis pipeline: Fast ML-based sentiment scoring via FinTwitBERT for all messages, followed by selective deep analysis via Claude API for high-signal messages. AnalyzerManager orchestrates the flow.

**Tech Stack:** Python 3.11+, transformers (HuggingFace), anthropic SDK, Pydantic v2, asyncio

---

## Task 0: Add Dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml` (if exists)

**Step 1: Add new dependencies to requirements.txt**

Add these lines to `requirements.txt`:
```
transformers>=4.36.0
torch>=2.0.0
anthropic>=0.40.0
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Dependencies install successfully

**Step 3: Verify imports work**

Run: `python -c "from transformers import pipeline; from anthropic import Anthropic; print('OK')"`
Expected: Prints "OK"

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add transformers and anthropic dependencies for Phase 2"
```

---

## Task 1: Create SentimentResult Model

**Files:**
- Create: `src/analyzers/__init__.py`
- Create: `src/analyzers/sentiment_result.py`
- Create: `tests/analyzers/__init__.py`
- Create: `tests/analyzers/test_sentiment_result.py`

**Step 1: Create analyzers package init**

Create `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""
```

Create `tests/analyzers/__init__.py`:
```python
"""Tests for analyzers package."""
```

**Step 2: Write the failing test**

Create `tests/analyzers/test_sentiment_result.py`:
```python
# tests/analyzers/test_sentiment_result.py
import pytest
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel


class TestSentimentResult:
    def test_create_bullish_result(self):
        result = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )
        assert result.label == SentimentLabel.BULLISH
        assert result.score == 0.92
        assert result.confidence == 0.88

    def test_create_bearish_result(self):
        result = SentimentResult(
            label=SentimentLabel.BEARISH,
            score=0.15,
            confidence=0.95,
        )
        assert result.label == SentimentLabel.BEARISH
        assert result.is_confident(min_confidence=0.7)

    def test_is_confident_above_threshold(self):
        result = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.85,
            confidence=0.80,
        )
        assert result.is_confident(min_confidence=0.7) is True
        assert result.is_confident(min_confidence=0.85) is False

    def test_neutral_label(self):
        result = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.60,
        )
        assert result.label == SentimentLabel.NEUTRAL
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/analyzers/test_sentiment_result.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.analyzers.sentiment_result'"

**Step 4: Write minimal implementation**

Create `src/analyzers/sentiment_result.py`:
```python
# src/analyzers/sentiment_result.py
from enum import Enum

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SentimentResult(BaseModel):
    """Result from sentiment analysis."""

    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Sentiment score 0-1, higher=more bullish")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in prediction")

    def is_confident(self, min_confidence: float = 0.7) -> bool:
        """Check if result meets minimum confidence threshold."""
        return self.confidence >= min_confidence
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/analyzers/test_sentiment_result.py -v`
Expected: 4 tests PASS

**Step 6: Update analyzers __init__.py**

Update `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel

__all__ = ["SentimentResult", "SentimentLabel"]
```

**Step 7: Commit**

```bash
git add src/analyzers/ tests/analyzers/
git commit -m "feat(analyzers): add SentimentResult model with label and confidence"
```

---

## Task 2: Create SentimentAnalyzer with FinTwitBERT

**Files:**
- Create: `src/analyzers/sentiment_analyzer.py`
- Create: `tests/analyzers/test_sentiment_analyzer.py`

**Step 1: Write the failing test**

Create `tests/analyzers/test_sentiment_analyzer.py`:
```python
# tests/analyzers/test_sentiment_analyzer.py
import pytest
from unittest.mock import MagicMock, patch
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.sentiment_result import SentimentLabel


class TestSentimentAnalyzer:
    @pytest.fixture
    def mock_pipeline(self):
        """Mock the transformers pipeline to avoid loading actual model."""
        with patch("src.analyzers.sentiment_analyzer.pipeline") as mock:
            mock_model = MagicMock()
            mock.return_value = mock_model
            yield mock_model

    def test_init_loads_model(self, mock_pipeline):
        with patch("src.analyzers.sentiment_analyzer.pipeline") as mock_pipe:
            mock_pipe.return_value = mock_pipeline
            analyzer = SentimentAnalyzer()
            mock_pipe.assert_called_once()

    def test_analyze_bullish_message(self, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "Bullish", "score": 0.92},
            {"label": "Bearish", "score": 0.05},
            {"label": "Neutral", "score": 0.03},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("🚨 Large $NVDA call sweep, very bullish!")

        assert result.label == SentimentLabel.BULLISH
        assert result.score == pytest.approx(0.92, rel=0.01)

    def test_analyze_bearish_message(self, mock_pipeline):
        mock_pipeline.return_value = [[
            {"label": "Bullish", "score": 0.10},
            {"label": "Bearish", "score": 0.85},
            {"label": "Neutral", "score": 0.05},
        ]]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("$AAPL looking weak, expecting pullback")

        assert result.label == SentimentLabel.BEARISH

    def test_analyze_batch(self, mock_pipeline):
        mock_pipeline.return_value = [
            [{"label": "Bullish", "score": 0.90}, {"label": "Bearish", "score": 0.05}, {"label": "Neutral", "score": 0.05}],
            [{"label": "Bearish", "score": 0.80}, {"label": "Bullish", "score": 0.10}, {"label": "Neutral", "score": 0.10}],
        ]

        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch(["bullish msg", "bearish msg"])

        assert len(results) == 2
        assert results[0].label == SentimentLabel.BULLISH
        assert results[1].label == SentimentLabel.BEARISH

    def test_analyze_empty_text_returns_neutral(self, mock_pipeline):
        with patch("src.analyzers.sentiment_analyzer.pipeline", return_value=mock_pipeline):
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("")

        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/analyzers/test_sentiment_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.analyzers.sentiment_analyzer'"

**Step 3: Write minimal implementation**

Create `src/analyzers/sentiment_analyzer.py`:
```python
# src/analyzers/sentiment_analyzer.py
from transformers import pipeline

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel


class SentimentAnalyzer:
    """Sentiment analyzer using FinTwitBERT model."""

    DEFAULT_MODEL = "StephanAkkerman/FinTwitBERT-sentiment"

    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        """Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name. Defaults to FinTwitBERT.
            batch_size: Batch size for inference.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,  # Return all labels with scores
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.

        Args:
            text: Text to analyze.

        Returns:
            SentimentResult with label, score, and confidence.
        """
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                confidence=0.0,
            )

        results = self._pipeline(text)
        return self._parse_result(results[0] if isinstance(results[0], list) else results)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of SentimentResult objects.
        """
        if not texts:
            return []

        # Filter empty texts, keep track of indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        # Run batch inference
        if valid_texts:
            batch_results = self._pipeline(valid_texts, batch_size=self.batch_size)
        else:
            batch_results = []

        # Build results list with neutral for empty texts
        results = [
            SentimentResult(label=SentimentLabel.NEUTRAL, score=0.5, confidence=0.0)
            for _ in texts
        ]

        for idx, batch_result in zip(valid_indices, batch_results):
            parsed = batch_result if isinstance(batch_result, list) else [batch_result]
            results[idx] = self._parse_result(parsed)

        return results

    def _parse_result(self, predictions: list[dict]) -> SentimentResult:
        """Parse model output to SentimentResult.

        Args:
            predictions: List of {label, score} dicts from model.

        Returns:
            SentimentResult.
        """
        # Find the label with highest score
        label_scores = {}
        for pred in predictions:
            label = pred["label"].lower()
            label_scores[label] = pred["score"]

        # Map to our labels
        bullish_score = label_scores.get("bullish", 0.0)
        bearish_score = label_scores.get("bearish", 0.0)
        neutral_score = label_scores.get("neutral", 0.0)

        # Determine label from highest score
        max_score = max(bullish_score, bearish_score, neutral_score)
        if bullish_score == max_score:
            label = SentimentLabel.BULLISH
            score = bullish_score
        elif bearish_score == max_score:
            label = SentimentLabel.BEARISH
            score = 1.0 - bearish_score  # Invert so 0 = very bearish
        else:
            label = SentimentLabel.NEUTRAL
            score = 0.5

        return SentimentResult(
            label=label,
            score=score if label == SentimentLabel.BULLISH else (1.0 - bearish_score if label == SentimentLabel.BEARISH else 0.5),
            confidence=max_score,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/analyzers/test_sentiment_analyzer.py -v`
Expected: 5 tests PASS

**Step 5: Update analyzers __init__.py**

Update `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.sentiment_analyzer import SentimentAnalyzer

__all__ = ["SentimentResult", "SentimentLabel", "SentimentAnalyzer"]
```

**Step 6: Commit**

```bash
git add src/analyzers/ tests/analyzers/
git commit -m "feat(analyzers): add SentimentAnalyzer with FinTwitBERT integration"
```

---

## Task 3: Create ClaudeAnalysisResult Model

**Files:**
- Create: `src/analyzers/claude_result.py`
- Create: `tests/analyzers/test_claude_result.py`

**Step 1: Write the failing test**

Create `tests/analyzers/test_claude_result.py`:
```python
# tests/analyzers/test_claude_result.py
import pytest
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel


class TestClaudeAnalysisResult:
    def test_create_result(self):
        result = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            risk_factors=["earnings_in_3_weeks"],
            context_summary="Large call sweep indicates institutional accumulation",
            recommendation="valid_catalyst",
            reasoning="Sweep size and timing suggest informed buying",
        )
        assert result.catalyst_type == CatalystType.INSTITUTIONAL_FLOW
        assert result.risk_level == RiskLevel.LOW
        assert len(result.risk_factors) == 1

    def test_catalyst_types(self):
        for catalyst in CatalystType:
            result = ClaudeAnalysisResult(
                catalyst_type=catalyst,
                catalyst_confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                context_summary="test",
                recommendation="test",
            )
            assert result.catalyst_type == catalyst

    def test_is_actionable(self):
        high_conf_result = ClaudeAnalysisResult(
            catalyst_type=CatalystType.EARNINGS,
            catalyst_confidence=0.80,
            risk_level=RiskLevel.MEDIUM,
            context_summary="test",
            recommendation="valid",
        )
        assert high_conf_result.is_actionable(min_confidence=0.7)

        low_conf_result = ClaudeAnalysisResult(
            catalyst_type=CatalystType.EARNINGS,
            catalyst_confidence=0.50,
            risk_level=RiskLevel.HIGH,
            context_summary="test",
            recommendation="skip",
        )
        assert not low_conf_result.is_actionable(min_confidence=0.7)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/analyzers/test_claude_result.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.analyzers.claude_result'"

**Step 3: Write minimal implementation**

Create `src/analyzers/claude_result.py`:
```python
# src/analyzers/claude_result.py
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CatalystType(str, Enum):
    """Types of catalysts detected in social messages."""
    INSTITUTIONAL_FLOW = "institutional_flow"
    EARNINGS = "earnings"
    BREAKING_NEWS = "breaking_news"
    TECHNICAL_BREAKOUT = "technical_breakout"
    SECTOR_ROTATION = "sector_rotation"
    INSIDER_ACTIVITY = "insider_activity"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    ANALYST_UPGRADE = "analyst_upgrade"
    SHORT_SQUEEZE = "short_squeeze"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk levels for trading opportunities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ClaudeAnalysisResult(BaseModel):
    """Result from Claude API deep analysis."""

    catalyst_type: CatalystType
    catalyst_confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)
    context_summary: str
    recommendation: str
    reasoning: Optional[str] = None

    def is_actionable(self, min_confidence: float = 0.7) -> bool:
        """Check if this analysis suggests an actionable opportunity.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            True if catalyst is confident enough and risk is acceptable.
        """
        return (
            self.catalyst_confidence >= min_confidence
            and self.risk_level != RiskLevel.EXTREME
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/analyzers/test_claude_result.py -v`
Expected: 3 tests PASS

**Step 5: Update analyzers __init__.py**

Update `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel

__all__ = [
    "SentimentResult",
    "SentimentLabel",
    "SentimentAnalyzer",
    "ClaudeAnalysisResult",
    "CatalystType",
    "RiskLevel",
]
```

**Step 6: Commit**

```bash
git add src/analyzers/ tests/analyzers/
git commit -m "feat(analyzers): add ClaudeAnalysisResult model with catalyst types"
```

---

## Task 4: Create ClaudeAnalyzer

**Files:**
- Create: `src/analyzers/claude_analyzer.py`
- Create: `tests/analyzers/test_claude_analyzer.py`

**Step 1: Write the failing test**

Create `tests/analyzers/test_claude_analyzer.py`:
```python
# tests/analyzers/test_claude_analyzer.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.claude_result import CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType
from datetime import datetime, timezone


class TestClaudeAnalyzer:
    @pytest.fixture
    def mock_anthropic(self):
        with patch("src.analyzers.claude_analyzer.Anthropic") as mock:
            yield mock

    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep $142 strike 2/21 exp, $2.4M premium",
            timestamp=datetime.now(timezone.utc),
            url="https://twitter.com/unusual_whales/status/123",
        )

    def test_init_creates_client(self, mock_anthropic):
        analyzer = ClaudeAnalyzer(api_key="test-key")
        mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_analyze_returns_result(self, mock_anthropic, sample_message):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "catalyst_type": "institutional_flow",
            "catalyst_confidence": 0.85,
            "risk_level": "low",
            "risk_factors": ["earnings_in_3_weeks"],
            "context_summary": "Large call sweep indicates institutional accumulation",
            "recommendation": "valid_catalyst",
            "reasoning": "Sweep size and timing suggest informed buying",
        }))]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        analyzer = ClaudeAnalyzer(api_key="test-key")
        result = analyzer.analyze(sample_message)

        assert result.catalyst_type == CatalystType.INSTITUTIONAL_FLOW
        assert result.catalyst_confidence == 0.85
        assert result.risk_level == RiskLevel.LOW
        assert "earnings_in_3_weeks" in result.risk_factors

    def test_analyze_handles_api_error(self, mock_anthropic, sample_message):
        mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

        analyzer = ClaudeAnalyzer(api_key="test-key")
        result = analyzer.analyze(sample_message)

        assert result.catalyst_type == CatalystType.UNKNOWN
        assert result.risk_level == RiskLevel.HIGH

    def test_rate_limiting(self, mock_anthropic, sample_message):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "catalyst_type": "institutional_flow",
            "catalyst_confidence": 0.85,
            "risk_level": "low",
            "risk_factors": [],
            "context_summary": "test",
            "recommendation": "valid",
        }))]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        analyzer = ClaudeAnalyzer(api_key="test-key", rate_limit_per_minute=60)
        assert analyzer.rate_limit_per_minute == 60
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/analyzers/test_claude_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.analyzers.claude_analyzer'"

**Step 3: Write minimal implementation**

Create `src/analyzers/claude_analyzer.py`:
```python
# src/analyzers/claude_analyzer.py
import json
import time
from typing import Optional

from anthropic import Anthropic

from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage


class ClaudeAnalyzer:
    """Deep analysis using Claude API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 1000

    ANALYSIS_PROMPT = '''Analyze this social media post about a stock trade opportunity.

Post:
Source: {source}
Author: {author}
Content: {content}
Tickers mentioned: {tickers}

Analyze and respond with a JSON object containing:
- catalyst_type: One of [institutional_flow, earnings, breaking_news, technical_breakout, sector_rotation, insider_activity, fda_approval, merger_acquisition, analyst_upgrade, short_squeeze, unknown]
- catalyst_confidence: Float 0-1 indicating confidence in catalyst identification
- risk_level: One of [low, medium, high, extreme]
- risk_factors: List of specific risk factors identified
- context_summary: Brief summary of the context and why this might be significant
- recommendation: Your recommendation (valid_catalyst, needs_verification, skip, or similar)
- reasoning: Brief explanation of your analysis

Respond ONLY with valid JSON, no other text.'''

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        rate_limit_per_minute: int = 20,
    ):
        """Initialize Claude analyzer.

        Args:
            api_key: Anthropic API key.
            model: Model to use. Defaults to claude-sonnet-4-20250514.
            max_tokens: Max response tokens.
            rate_limit_per_minute: Rate limit for API calls.
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_call_time: Optional[float] = None

    def analyze(self, message: SocialMessage) -> ClaudeAnalysisResult:
        """Analyze a social message for catalyst and risk assessment.

        Args:
            message: SocialMessage to analyze.

        Returns:
            ClaudeAnalysisResult with analysis.
        """
        self._enforce_rate_limit()

        try:
            tickers = message.extract_tickers()
            prompt = self.ANALYSIS_PROMPT.format(
                source=message.source.value,
                author=message.author,
                content=message.content,
                tickers=", ".join(tickers) if tickers else "None detected",
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            result_data = json.loads(result_text)

            return ClaudeAnalysisResult(
                catalyst_type=CatalystType(result_data["catalyst_type"]),
                catalyst_confidence=result_data["catalyst_confidence"],
                risk_level=RiskLevel(result_data["risk_level"]),
                risk_factors=result_data.get("risk_factors", []),
                context_summary=result_data["context_summary"],
                recommendation=result_data["recommendation"],
                reasoning=result_data.get("reasoning"),
            )

        except Exception:
            # Return unknown/high-risk result on error
            return ClaudeAnalysisResult(
                catalyst_type=CatalystType.UNKNOWN,
                catalyst_confidence=0.0,
                risk_level=RiskLevel.HIGH,
                risk_factors=["analysis_failed"],
                context_summary="Analysis failed",
                recommendation="skip",
                reasoning="API error or parsing failure",
            )

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        if self._last_call_time is not None:
            min_interval = 60.0 / self.rate_limit_per_minute
            elapsed = time.time() - self._last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/analyzers/test_claude_analyzer.py -v`
Expected: 4 tests PASS

**Step 5: Update analyzers __init__.py**

Update `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.analyzers.claude_analyzer import ClaudeAnalyzer

__all__ = [
    "SentimentResult",
    "SentimentLabel",
    "SentimentAnalyzer",
    "ClaudeAnalysisResult",
    "CatalystType",
    "RiskLevel",
    "ClaudeAnalyzer",
]
```

**Step 6: Commit**

```bash
git add src/analyzers/ tests/analyzers/
git commit -m "feat(analyzers): add ClaudeAnalyzer for deep catalyst analysis"
```

---

## Task 5: Create AnalyzedMessage Model

**Files:**
- Create: `src/analyzers/analyzed_message.py`
- Create: `tests/analyzers/test_analyzed_message.py`

**Step 1: Write the failing test**

Create `tests/analyzers/test_analyzed_message.py`:
```python
# tests/analyzers/test_analyzed_message.py
import pytest
from datetime import datetime, timezone
from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType


class TestAnalyzedMessage:
    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_sentiment(self):
        return SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )

    def test_create_with_sentiment_only(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.message == sample_message
        assert analyzed.sentiment == sample_sentiment
        assert analyzed.deep_analysis is None

    def test_create_with_deep_analysis(self, sample_message, sample_sentiment):
        deep = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            context_summary="test",
            recommendation="valid",
        )
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
            deep_analysis=deep,
        )
        assert analyzed.deep_analysis == deep

    def test_is_high_signal(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        # High sentiment confidence + bullish = high signal
        assert analyzed.is_high_signal(min_sentiment_confidence=0.7)

    def test_requires_deep_analysis(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        # Bullish with high confidence should require deep analysis
        assert analyzed.requires_deep_analysis(min_sentiment_confidence=0.7)

    def test_get_tickers(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.get_tickers() == ["NVDA"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/analyzers/test_analyzed_message.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.analyzers.analyzed_message'"

**Step 3: Write minimal implementation**

Create `src/analyzers/analyzed_message.py`:
```python
# src/analyzers/analyzed_message.py
from typing import Optional

from pydantic import BaseModel

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult
from src.models.social_message import SocialMessage


class AnalyzedMessage(BaseModel):
    """A social message with sentiment and optional deep analysis."""

    message: SocialMessage
    sentiment: SentimentResult
    deep_analysis: Optional[ClaudeAnalysisResult] = None

    def is_high_signal(self, min_sentiment_confidence: float = 0.7) -> bool:
        """Check if this message is a high-signal opportunity.

        Args:
            min_sentiment_confidence: Minimum confidence threshold.

        Returns:
            True if sentiment is confident and not neutral.
        """
        return (
            self.sentiment.is_confident(min_sentiment_confidence)
            and self.sentiment.label != SentimentLabel.NEUTRAL
        )

    def requires_deep_analysis(self, min_sentiment_confidence: float = 0.7) -> bool:
        """Check if this message should get deep analysis.

        Args:
            min_sentiment_confidence: Minimum confidence for triggering deep analysis.

        Returns:
            True if sentiment is confident enough to warrant deeper analysis.
        """
        return (
            self.deep_analysis is None
            and self.sentiment.is_confident(min_sentiment_confidence)
            and self.sentiment.label != SentimentLabel.NEUTRAL
        )

    def get_tickers(self, exclude_crypto: bool = True) -> list[str]:
        """Get tickers from the underlying message.

        Args:
            exclude_crypto: Whether to exclude crypto tickers.

        Returns:
            List of ticker symbols.
        """
        return self.message.extract_tickers(exclude_crypto=exclude_crypto)

    class Config:
        arbitrary_types_allowed = True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/analyzers/test_analyzed_message.py -v`
Expected: 5 tests PASS

**Step 5: Update analyzers __init__.py**

Update `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.analyzed_message import AnalyzedMessage

__all__ = [
    "SentimentResult",
    "SentimentLabel",
    "SentimentAnalyzer",
    "ClaudeAnalysisResult",
    "CatalystType",
    "RiskLevel",
    "ClaudeAnalyzer",
    "AnalyzedMessage",
]
```

**Step 6: Commit**

```bash
git add src/analyzers/ tests/analyzers/
git commit -m "feat(analyzers): add AnalyzedMessage model combining sentiment and deep analysis"
```

---

## Task 6: Create AnalyzerManager Pipeline

**Files:**
- Create: `src/analyzers/analyzer_manager.py`
- Create: `tests/analyzers/test_analyzer_manager.py`

**Step 1: Write the failing test**

Create `tests/analyzers/test_analyzer_manager.py`:
```python
# tests/analyzers/test_analyzer_manager.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
from src.analyzers.analyzer_manager import AnalyzerManager
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType


class TestAnalyzerManager:
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        mock = MagicMock()
        mock.analyze.return_value = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.90,
            confidence=0.85,
        )
        mock.analyze_batch.return_value = [
            SentimentResult(label=SentimentLabel.BULLISH, score=0.90, confidence=0.85),
            SentimentResult(label=SentimentLabel.NEUTRAL, score=0.50, confidence=0.60),
        ]
        return mock

    @pytest.fixture
    def mock_claude_analyzer(self):
        mock = MagicMock()
        mock.analyze.return_value = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            context_summary="Large sweep detected",
            recommendation="valid",
        )
        return mock

    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
        )

    def test_init(self, mock_sentiment_analyzer, mock_claude_analyzer):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
        )
        assert manager.sentiment_analyzer == mock_sentiment_analyzer
        assert manager.claude_analyzer == mock_claude_analyzer

    def test_analyze_single_message(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
        )
        result = manager.analyze(sample_message)

        assert result.sentiment.label == SentimentLabel.BULLISH
        mock_sentiment_analyzer.analyze.assert_called_once_with(sample_message.content)

    def test_analyze_triggers_deep_analysis_for_high_signal(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )
        result = manager.analyze(sample_message)

        # Should trigger deep analysis for bullish message with high confidence
        assert result.deep_analysis is not None
        mock_claude_analyzer.analyze.assert_called_once()

    def test_analyze_skips_deep_analysis_for_low_confidence(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.60,
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )
        result = manager.analyze(sample_message)

        # Should NOT trigger deep analysis for neutral/low confidence
        assert result.deep_analysis is None
        mock_claude_analyzer.analyze.assert_not_called()

    def test_analyze_batch(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        messages = [sample_message, sample_message]

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            enable_deep_analysis=False,  # Disable for batch test simplicity
        )
        results = manager.analyze_batch(messages)

        assert len(results) == 2
        mock_sentiment_analyzer.analyze_batch.assert_called_once()

    def test_disabled_deep_analysis(
        self, mock_sentiment_analyzer, mock_claude_analyzer, sample_message
    ):
        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            enable_deep_analysis=False,
        )
        result = manager.analyze(sample_message)

        assert result.deep_analysis is None
        mock_claude_analyzer.analyze.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/analyzers/test_analyzer_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.analyzers.analyzer_manager'"

**Step 3: Write minimal implementation**

Create `src/analyzers/analyzer_manager.py`:
```python
# src/analyzers/analyzer_manager.py
from typing import Optional

from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.analyzed_message import AnalyzedMessage
from src.models.social_message import SocialMessage


class AnalyzerManager:
    """Orchestrates the analysis pipeline: sentiment → deep analysis."""

    def __init__(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        claude_analyzer: Optional[ClaudeAnalyzer] = None,
        min_sentiment_confidence: float = 0.7,
        enable_deep_analysis: bool = True,
    ):
        """Initialize the analyzer manager.

        Args:
            sentiment_analyzer: Sentiment analyzer instance.
            claude_analyzer: Claude analyzer for deep analysis (optional).
            min_sentiment_confidence: Minimum confidence to trigger deep analysis.
            enable_deep_analysis: Whether to enable Claude deep analysis.
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.claude_analyzer = claude_analyzer
        self.min_sentiment_confidence = min_sentiment_confidence
        self.enable_deep_analysis = enable_deep_analysis and claude_analyzer is not None

    def analyze(self, message: SocialMessage) -> AnalyzedMessage:
        """Analyze a single social message.

        Pipeline:
        1. Run sentiment analysis
        2. If high signal and deep analysis enabled, run Claude analysis

        Args:
            message: Social message to analyze.

        Returns:
            AnalyzedMessage with sentiment and optional deep analysis.
        """
        # Step 1: Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze(message.content)

        # Create initial analyzed message
        analyzed = AnalyzedMessage(
            message=message,
            sentiment=sentiment,
        )

        # Step 2: Deep analysis if warranted
        if self.enable_deep_analysis and analyzed.requires_deep_analysis(
            self.min_sentiment_confidence
        ):
            deep_analysis = self.claude_analyzer.analyze(message)
            analyzed = AnalyzedMessage(
                message=message,
                sentiment=sentiment,
                deep_analysis=deep_analysis,
            )

        return analyzed

    def analyze_batch(self, messages: list[SocialMessage]) -> list[AnalyzedMessage]:
        """Analyze multiple messages in batch.

        Uses batch sentiment analysis for efficiency, then selectively
        applies deep analysis to high-signal messages.

        Args:
            messages: List of social messages.

        Returns:
            List of AnalyzedMessage objects.
        """
        if not messages:
            return []

        # Batch sentiment analysis
        contents = [m.content for m in messages]
        sentiments = self.sentiment_analyzer.analyze_batch(contents)

        # Build results and selectively apply deep analysis
        results = []
        for message, sentiment in zip(messages, sentiments):
            analyzed = AnalyzedMessage(
                message=message,
                sentiment=sentiment,
            )

            if self.enable_deep_analysis and analyzed.requires_deep_analysis(
                self.min_sentiment_confidence
            ):
                deep_analysis = self.claude_analyzer.analyze(message)
                analyzed = AnalyzedMessage(
                    message=message,
                    sentiment=sentiment,
                    deep_analysis=deep_analysis,
                )

            results.append(analyzed)

        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/analyzers/test_analyzer_manager.py -v`
Expected: 6 tests PASS

**Step 5: Update analyzers __init__.py**

Update `src/analyzers/__init__.py`:
```python
"""Analyzers package for sentiment and deep analysis."""

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.analyzer_manager import AnalyzerManager

__all__ = [
    "SentimentResult",
    "SentimentLabel",
    "SentimentAnalyzer",
    "ClaudeAnalysisResult",
    "CatalystType",
    "RiskLevel",
    "ClaudeAnalyzer",
    "AnalyzedMessage",
    "AnalyzerManager",
]
```

**Step 6: Commit**

```bash
git add src/analyzers/ tests/analyzers/
git commit -m "feat(analyzers): add AnalyzerManager pipeline orchestrator"
```

---

## Task 7: Update Settings for Analyzer Configuration

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Create: `tests/config/test_analyzer_settings.py`

**Step 1: Write the failing test**

Create `tests/config/test_analyzer_settings.py`:
```python
# tests/config/test_analyzer_settings.py
import pytest
from src.config.settings import Settings


class TestAnalyzerSettings:
    def test_settings_has_analyzer_config(self):
        settings = Settings()
        assert hasattr(settings, 'analyzers')

    def test_sentiment_analyzer_defaults(self):
        settings = Settings()
        assert settings.analyzers.sentiment.model == "StephanAkkerman/FinTwitBERT-sentiment"
        assert settings.analyzers.sentiment.batch_size == 32
        assert settings.analyzers.sentiment.min_confidence == 0.7

    def test_claude_analyzer_defaults(self):
        settings = Settings()
        assert settings.analyzers.claude.enabled is True
        assert settings.analyzers.claude.model == "claude-sonnet-4-20250514"
        assert settings.analyzers.claude.max_tokens == 1000
        assert settings.analyzers.claude.rate_limit_per_minute == 20
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_analyzer_settings.py -v`
Expected: FAIL with AttributeError (no 'analyzers' attribute)

**Step 3: Update settings.py**

Add to `src/config/settings.py` (add these new Pydantic models and update Settings class):

```python
# Add these new classes to src/config/settings.py

class SentimentAnalyzerSettings(BaseModel):
    """Settings for sentiment analyzer."""
    model: str = "StephanAkkerman/FinTwitBERT-sentiment"
    batch_size: int = 32
    min_confidence: float = 0.7


class ClaudeAnalyzerSettings(BaseModel):
    """Settings for Claude analyzer."""
    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1000
    rate_limit_per_minute: int = 20
    use_for: list[str] = Field(default_factory=lambda: [
        "catalyst_classification",
        "risk_assessment",
        "context_analysis",
    ])


class AnalyzersSettings(BaseModel):
    """Settings for all analyzers."""
    sentiment: SentimentAnalyzerSettings = Field(default_factory=SentimentAnalyzerSettings)
    claude: ClaudeAnalyzerSettings = Field(default_factory=ClaudeAnalyzerSettings)


# Update Settings class to include analyzers
class Settings(BaseModel):
    # ... existing fields ...
    analyzers: AnalyzersSettings = Field(default_factory=AnalyzersSettings)
```

**Step 4: Update config/settings.yaml**

Add analyzers section to `config/settings.yaml`:

```yaml
# CAPA 2: Análisis
analyzers:
  sentiment:
    model: "StephanAkkerman/FinTwitBERT-sentiment"
    batch_size: 32
    min_confidence: 0.7

  claude:
    enabled: true
    model: "claude-sonnet-4-20250514"
    max_tokens: 1000
    rate_limit_per_minute: 20
    use_for:
      - catalyst_classification
      - risk_assessment
      - context_analysis
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/config/test_analyzer_settings.py -v`
Expected: 3 tests PASS

**Step 6: Run all existing tests to ensure no regressions**

Run: `pytest -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/config/settings.py config/settings.yaml tests/config/
git commit -m "feat(config): add analyzer settings for sentiment and Claude"
```

---

## Task 8: Integration Test - Full Pipeline

**Files:**
- Create: `tests/integration/test_analyzer_pipeline.py`

**Step 1: Write the integration test**

Create `tests/integration/test_analyzer_pipeline.py`:
```python
# tests/integration/test_analyzer_pipeline.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.models.social_message import SocialMessage, SourceType
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.analyzers.analyzer_manager import AnalyzerManager


class TestAnalyzerPipelineIntegration:
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_claude_analyzer(self):
        mock = MagicMock()
        return mock

    def test_full_pipeline_bullish_flow(
        self, mock_sentiment_analyzer, mock_claude_analyzer
    ):
        """Test full pipeline: bullish message → sentiment → deep analysis."""
        # Setup
        message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep $142 strike, $2.4M premium",
            timestamp=datetime.now(timezone.utc),
        )

        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )

        mock_claude_analyzer.analyze.return_value = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            risk_factors=["earnings_in_3_weeks"],
            context_summary="Large call sweep indicates institutional accumulation",
            recommendation="valid_catalyst",
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )

        # Execute
        result = manager.analyze(message)

        # Verify
        assert result.sentiment.label == SentimentLabel.BULLISH
        assert result.sentiment.confidence >= 0.7
        assert result.deep_analysis is not None
        assert result.deep_analysis.catalyst_type == CatalystType.INSTITUTIONAL_FLOW
        assert result.is_high_signal()

    def test_full_pipeline_neutral_flow(
        self, mock_sentiment_analyzer, mock_claude_analyzer
    ):
        """Test pipeline: neutral message → sentiment only, no deep analysis."""
        message = SocialMessage(
            source=SourceType.REDDIT,
            source_id="456",
            author="random_user",
            content="What do you guys think about $AAPL?",
            timestamp=datetime.now(timezone.utc),
            subreddit="stocks",
        )

        mock_sentiment_analyzer.analyze.return_value = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.65,
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )

        result = manager.analyze(message)

        # Verify - should NOT trigger deep analysis
        assert result.sentiment.label == SentimentLabel.NEUTRAL
        assert result.deep_analysis is None
        mock_claude_analyzer.analyze.assert_not_called()

    def test_batch_processing(self, mock_sentiment_analyzer, mock_claude_analyzer):
        """Test batch processing of multiple messages."""
        messages = [
            SocialMessage(
                source=SourceType.TWITTER,
                source_id="1",
                author="trader1",
                content="$NVDA looking strong!",
                timestamp=datetime.now(timezone.utc),
            ),
            SocialMessage(
                source=SourceType.STOCKTWITS,
                source_id="2",
                author="trader2",
                content="$AAPL might pull back",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        mock_sentiment_analyzer.analyze_batch.return_value = [
            SentimentResult(label=SentimentLabel.BULLISH, score=0.85, confidence=0.80),
            SentimentResult(label=SentimentLabel.BEARISH, score=0.25, confidence=0.75),
        ]

        mock_claude_analyzer.analyze.return_value = ClaudeAnalysisResult(
            catalyst_type=CatalystType.TECHNICAL_BREAKOUT,
            catalyst_confidence=0.70,
            risk_level=RiskLevel.MEDIUM,
            context_summary="test",
            recommendation="valid",
        )

        manager = AnalyzerManager(
            sentiment_analyzer=mock_sentiment_analyzer,
            claude_analyzer=mock_claude_analyzer,
            min_sentiment_confidence=0.7,
            enable_deep_analysis=True,
        )

        results = manager.analyze_batch(messages)

        assert len(results) == 2
        # Both should have deep analysis (both above confidence threshold)
        assert results[0].deep_analysis is not None
        assert results[1].deep_analysis is not None
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_analyzer_pipeline.py -v`
Expected: 3 tests PASS

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: add integration tests for analyzer pipeline"
```

---

## Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest -v --tb=short`
Expected: All tests pass (Phase 1 + Phase 2)

**Step 2: Check test coverage**

Run: `pytest --cov=src --cov-report=term-missing`
Expected: Coverage ≥ 85%

**Step 3: Final commit if needed**

```bash
git status
# If any uncommitted changes:
git add .
git commit -m "chore: Phase 2 complete - sentiment analysis pipeline"
```

---

## Summary

Phase 2 implements CAPA 2 (Processing & Analysis) with:

1. **SentimentResult** - Model for sentiment analysis output
2. **SentimentAnalyzer** - FinTwitBERT integration for fast sentiment scoring
3. **ClaudeAnalysisResult** - Model for deep analysis output
4. **ClaudeAnalyzer** - Claude API integration for catalyst/risk analysis
5. **AnalyzedMessage** - Combined model for processed messages
6. **AnalyzerManager** - Pipeline orchestrator

The pipeline flow:
```
SocialMessage → SentimentAnalyzer → AnalyzedMessage
                                          ↓
                        (if high signal) ClaudeAnalyzer
                                          ↓
                                  AnalyzedMessage with deep_analysis
```

Next phases will build on this:
- Phase 3: Technical validation (CAPA 3)
- Phase 4: Scoring engine (CAPA 4)
- Phase 5: Risk management (CAPA 5)
