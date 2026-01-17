# Phase 3: Technical Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add technical validation layer (CAPA 3) that uses market data and indicators to veto dangerous trading signals from sentiment analysis.

**Architecture:** TechnicalValidator receives AnalyzedMessage from Phase 2, fetches market data via Alpaca, computes momentum indicators (RSI, MACD, Stochastic, ADX), applies veto rules to block dangerous entries, and outputs ValidatedSignal with pass/veto/warn status.

**Tech Stack:** Alpaca API (market data), pandas-ta (indicators), Pydantic (models), pytest (TDD)

---

## Task 1: Data Models

**Files:**
- Create: `src/validators/__init__.py`
- Create: `src/validators/models.py`
- Test: `tests/validators/test_models.py`

**Step 1: Write the failing test**

```python
# tests/validators/test_models.py
import pytest
from src.validators.models import (
    TechnicalIndicators,
    OptionsFlowData,
    ValidationStatus,
    TechnicalValidation,
    ValidatedSignal,
)


class TestTechnicalIndicators:
    def test_creation_with_all_fields(self):
        indicators = TechnicalIndicators(
            rsi=65.5,
            macd_histogram=0.25,
            macd_trend="rising",
            stochastic_k=75.0,
            stochastic_d=70.0,
            adx=28.5,
            rsi_divergence=None,
            macd_divergence="bullish",
        )
        assert indicators.rsi == 65.5
        assert indicators.macd_trend == "rising"
        assert indicators.macd_divergence == "bullish"


class TestOptionsFlowData:
    def test_creation(self):
        options = OptionsFlowData(
            volume_ratio=2.5,
            iv_rank=45.0,
            put_call_ratio=0.8,
            unusual_activity=True,
        )
        assert options.volume_ratio == 2.5
        assert options.unusual_activity is True


class TestValidationStatus:
    def test_enum_values(self):
        assert ValidationStatus.PASS.value == "pass"
        assert ValidationStatus.VETO.value == "veto"
        assert ValidationStatus.WARN.value == "warn"


class TestTechnicalValidation:
    def test_creation_pass_status(self):
        indicators = TechnicalIndicators(
            rsi=50.0,
            macd_histogram=0.1,
            macd_trend="rising",
            stochastic_k=50.0,
            stochastic_d=50.0,
            adx=25.0,
        )
        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=indicators,
            options_flow=None,
            veto_reasons=[],
            warnings=[],
            confidence_modifier=1.0,
        )
        assert validation.status == ValidationStatus.PASS
        assert validation.confidence_modifier == 1.0

    def test_creation_veto_status(self):
        indicators = TechnicalIndicators(
            rsi=75.0,
            macd_histogram=-0.1,
            macd_trend="falling",
            stochastic_k=85.0,
            stochastic_d=80.0,
            adx=30.0,
        )
        validation = TechnicalValidation(
            status=ValidationStatus.VETO,
            indicators=indicators,
            options_flow=None,
            veto_reasons=["RSI overbought at 75.0"],
            warnings=[],
            confidence_modifier=0.5,
        )
        assert validation.status == ValidationStatus.VETO
        assert len(validation.veto_reasons) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.validators'"

**Step 3: Write minimal implementation**

```python
# src/validators/__init__.py
"""Technical validation module for CAPA 3."""

from .models import (
    TechnicalIndicators,
    OptionsFlowData,
    ValidationStatus,
    TechnicalValidation,
    ValidatedSignal,
)

__all__ = [
    "TechnicalIndicators",
    "OptionsFlowData",
    "ValidationStatus",
    "TechnicalValidation",
    "ValidatedSignal",
]
```

```python
# src/validators/models.py
"""Data models for technical validation."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.analyzers.analyzer_manager import AnalyzedMessage


class ValidationStatus(Enum):
    """Status of technical validation."""

    PASS = "pass"
    VETO = "veto"
    WARN = "warn"


@dataclass
class TechnicalIndicators:
    """Technical indicator values."""

    rsi: float
    macd_histogram: float
    macd_trend: str  # "rising" | "falling" | "flat"
    stochastic_k: float
    stochastic_d: float
    adx: float
    rsi_divergence: Optional[str] = None  # "bullish" | "bearish" | None
    macd_divergence: Optional[str] = None


@dataclass
class OptionsFlowData:
    """Options flow validation data."""

    volume_ratio: float  # current / average (>2 = spike)
    iv_rank: float  # 0-100 percentile
    put_call_ratio: float
    unusual_activity: bool


@dataclass
class TechnicalValidation:
    """Result of technical validation."""

    status: ValidationStatus
    indicators: TechnicalIndicators
    options_flow: Optional[OptionsFlowData]
    veto_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence_modifier: float = 1.0


@dataclass
class ValidatedSignal:
    """Final validated signal combining sentiment and technical analysis."""

    message: AnalyzedMessage
    validation: TechnicalValidation

    def should_trade(self) -> bool:
        """Returns True if signal passed technical validation."""
        return self.validation.status != ValidationStatus.VETO
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/ tests/validators/
git commit -m "feat(validators): add data models for technical validation"
```

---

## Task 2: Indicator Engine - RSI

**Files:**
- Create: `src/validators/indicator_engine.py`
- Test: `tests/validators/test_indicator_engine.py`

**Step 1: Write the failing test**

```python
# tests/validators/test_indicator_engine.py
import pytest
import pandas as pd
import numpy as np
from src.validators.indicator_engine import IndicatorEngine


class TestIndicatorEngineRSI:
    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data with known RSI behavior."""
        # Uptrend prices should give RSI > 50
        np.random.seed(42)
        prices = [100.0]
        for _ in range(49):
            prices.append(prices[-1] * (1 + np.random.uniform(0.001, 0.02)))
        return pd.Series(prices)

    @pytest.fixture
    def downtrend_prices(self):
        """Generate downtrend price data."""
        np.random.seed(42)
        prices = [100.0]
        for _ in range(49):
            prices.append(prices[-1] * (1 - np.random.uniform(0.001, 0.02)))
        return pd.Series(prices)

    def test_rsi_uptrend_above_50(self, engine, sample_prices):
        rsi = engine.calculate_rsi(sample_prices, period=14)
        assert rsi > 50

    def test_rsi_downtrend_below_50(self, engine, downtrend_prices):
        rsi = engine.calculate_rsi(downtrend_prices, period=14)
        assert rsi < 50

    def test_rsi_range(self, engine, sample_prices):
        rsi = engine.calculate_rsi(sample_prices, period=14)
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self, engine):
        short_prices = pd.Series([100.0, 101.0, 102.0])
        rsi = engine.calculate_rsi(short_prices, period=14)
        assert rsi == 50.0  # Default neutral value
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineRSI -v`
Expected: FAIL with "ModuleNotFoundError" or "AttributeError"

**Step 3: Write minimal implementation**

```python
# src/validators/indicator_engine.py
"""Technical indicator calculations."""
import pandas as pd
import pandas_ta as ta
from typing import Optional


class IndicatorEngine:
    """Calculates technical indicators from price data."""

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            Current RSI value (0-100), or 50.0 if insufficient data
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral default

        rsi = ta.rsi(prices, length=period)
        if rsi is None or rsi.empty or pd.isna(rsi.iloc[-1]):
            return 50.0

        return float(rsi.iloc[-1])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineRSI -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/indicator_engine.py tests/validators/test_indicator_engine.py
git commit -m "feat(validators): add RSI calculation to indicator engine"
```

---

## Task 3: Indicator Engine - MACD

**Files:**
- Modify: `src/validators/indicator_engine.py`
- Modify: `tests/validators/test_indicator_engine.py`

**Step 1: Write the failing test**

```python
# Add to tests/validators/test_indicator_engine.py

class TestIndicatorEngineMACD:
    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    @pytest.fixture
    def uptrend_prices(self):
        """Strong uptrend for positive MACD."""
        prices = [100.0]
        for i in range(59):
            prices.append(prices[-1] * 1.01)
        return pd.Series(prices)

    def test_macd_histogram_positive_uptrend(self, engine, uptrend_prices):
        histogram, trend = engine.calculate_macd(uptrend_prices)
        assert histogram > 0

    def test_macd_trend_rising(self, engine, uptrend_prices):
        histogram, trend = engine.calculate_macd(uptrend_prices)
        assert trend in ["rising", "falling", "flat"]

    def test_macd_insufficient_data(self, engine):
        short_prices = pd.Series([100.0] * 10)
        histogram, trend = engine.calculate_macd(short_prices)
        assert histogram == 0.0
        assert trend == "flat"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineMACD -v`
Expected: FAIL with "AttributeError: 'IndicatorEngine' object has no attribute 'calculate_macd'"

**Step 3: Write minimal implementation**

```python
# Add to src/validators/indicator_engine.py

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, str]:
        """
        Calculate MACD histogram and trend.

        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Tuple of (histogram_value, trend_direction)
            trend_direction is "rising", "falling", or "flat"
        """
        min_length = slow + signal
        if len(prices) < min_length:
            return 0.0, "flat"

        macd = ta.macd(prices, fast=fast, slow=slow, signal=signal)
        if macd is None or macd.empty:
            return 0.0, "flat"

        hist_col = f"MACDh_{fast}_{slow}_{signal}"
        if hist_col not in macd.columns:
            return 0.0, "flat"

        histogram = macd[hist_col]
        current = histogram.iloc[-1]

        if pd.isna(current):
            return 0.0, "flat"

        # Determine trend from last 3 values
        if len(histogram) >= 3:
            recent = histogram.iloc[-3:].dropna()
            if len(recent) >= 3:
                if recent.iloc[-1] > recent.iloc[-2] > recent.iloc[-3]:
                    trend = "rising"
                elif recent.iloc[-1] < recent.iloc[-2] < recent.iloc[-3]:
                    trend = "falling"
                else:
                    trend = "flat"
            else:
                trend = "flat"
        else:
            trend = "flat"

        return float(current), trend
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineMACD -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/indicator_engine.py tests/validators/test_indicator_engine.py
git commit -m "feat(validators): add MACD calculation to indicator engine"
```

---

## Task 4: Indicator Engine - Stochastic & ADX

**Files:**
- Modify: `src/validators/indicator_engine.py`
- Modify: `tests/validators/test_indicator_engine.py`

**Step 1: Write the failing test**

```python
# Add to tests/validators/test_indicator_engine.py

class TestIndicatorEngineStochastic:
    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    @pytest.fixture
    def ohlc_data(self):
        """Generate OHLC data for stochastic calculation."""
        np.random.seed(42)
        n = 50
        close = [100.0]
        for _ in range(n - 1):
            close.append(close[-1] * (1 + np.random.uniform(-0.02, 0.02)))
        close = pd.Series(close)
        high = close * 1.01
        low = close * 0.99
        return high, low, close

    def test_stochastic_range(self, engine, ohlc_data):
        high, low, close = ohlc_data
        k, d = engine.calculate_stochastic(high, low, close)
        assert 0 <= k <= 100
        assert 0 <= d <= 100

    def test_stochastic_insufficient_data(self, engine):
        short = pd.Series([100.0] * 5)
        k, d = engine.calculate_stochastic(short, short, short)
        assert k == 50.0
        assert d == 50.0


class TestIndicatorEngineADX:
    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    @pytest.fixture
    def trending_ohlc(self):
        """Strong trend data for ADX."""
        n = 50
        close = pd.Series([100.0 + i * 2 for i in range(n)])
        high = close + 1
        low = close - 1
        return high, low, close

    def test_adx_range(self, engine, trending_ohlc):
        high, low, close = trending_ohlc
        adx = engine.calculate_adx(high, low, close)
        assert 0 <= adx <= 100

    def test_adx_insufficient_data(self, engine):
        short = pd.Series([100.0] * 5)
        adx = engine.calculate_adx(short, short, short)
        assert adx == 0.0  # No trend detectable
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineStochastic -v`
Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineADX -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

```python
# Add to src/validators/indicator_engine.py

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[float, float]:
        """
        Calculate Stochastic %K and %D.

        Returns:
            Tuple of (stochastic_k, stochastic_d), or (50.0, 50.0) if insufficient data
        """
        if len(close) < k_period + d_period:
            return 50.0, 50.0

        stoch = ta.stoch(high, low, close, k=k_period, d=d_period)
        if stoch is None or stoch.empty:
            return 50.0, 50.0

        k_col = f"STOCHk_{k_period}_{d_period}_3"
        d_col = f"STOCHd_{k_period}_{d_period}_3"

        k_val = stoch[k_col].iloc[-1] if k_col in stoch.columns else 50.0
        d_val = stoch[d_col].iloc[-1] if d_col in stoch.columns else 50.0

        return (
            float(k_val) if not pd.isna(k_val) else 50.0,
            float(d_val) if not pd.isna(d_val) else 50.0,
        )

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        """
        Calculate ADX (Average Directional Index).

        Returns:
            ADX value (0-100), or 0.0 if insufficient data
        """
        if len(close) < period * 2:
            return 0.0

        adx = ta.adx(high, low, close, length=period)
        if adx is None or adx.empty:
            return 0.0

        adx_col = f"ADX_{period}"
        if adx_col not in adx.columns:
            return 0.0

        value = adx[adx_col].iloc[-1]
        return float(value) if not pd.isna(value) else 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_indicator_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/indicator_engine.py tests/validators/test_indicator_engine.py
git commit -m "feat(validators): add Stochastic and ADX to indicator engine"
```

---

## Task 5: Indicator Engine - Full Calculation Method

**Files:**
- Modify: `src/validators/indicator_engine.py`
- Modify: `tests/validators/test_indicator_engine.py`

**Step 1: Write the failing test**

```python
# Add to tests/validators/test_indicator_engine.py
from src.validators.models import TechnicalIndicators


class TestIndicatorEngineCalculateAll:
    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    @pytest.fixture
    def ohlc_dataframe(self):
        """Generate complete OHLC DataFrame."""
        np.random.seed(42)
        n = 60
        close = [100.0]
        for _ in range(n - 1):
            close.append(close[-1] * (1 + np.random.uniform(-0.015, 0.02)))
        close = pd.Series(close)
        return pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        })

    def test_calculate_all_returns_indicators(self, engine, ohlc_dataframe):
        indicators = engine.calculate_all(ohlc_dataframe)
        assert isinstance(indicators, TechnicalIndicators)
        assert 0 <= indicators.rsi <= 100
        assert indicators.macd_trend in ["rising", "falling", "flat"]
        assert 0 <= indicators.adx <= 100

    def test_calculate_all_with_insufficient_data(self, engine):
        short_df = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
        })
        indicators = engine.calculate_all(short_df)
        assert indicators.rsi == 50.0  # Neutral defaults
        assert indicators.adx == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineCalculateAll -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Add to src/validators/indicator_engine.py
from src.validators.models import TechnicalIndicators


    def calculate_all(self, ohlc: pd.DataFrame) -> TechnicalIndicators:
        """
        Calculate all technical indicators from OHLC data.

        Args:
            ohlc: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            TechnicalIndicators with all calculated values
        """
        close = ohlc["close"]
        high = ohlc["high"]
        low = ohlc["low"]

        rsi = self.calculate_rsi(close)
        macd_hist, macd_trend = self.calculate_macd(close)
        stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
        adx = self.calculate_adx(high, low, close)

        # Divergence detection (simplified)
        rsi_divergence = self._detect_rsi_divergence(close, rsi)
        macd_divergence = self._detect_macd_divergence(close, macd_hist)

        return TechnicalIndicators(
            rsi=rsi,
            macd_histogram=macd_hist,
            macd_trend=macd_trend,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            adx=adx,
            rsi_divergence=rsi_divergence,
            macd_divergence=macd_divergence,
        )

    def _detect_rsi_divergence(
        self, prices: pd.Series, current_rsi: float
    ) -> Optional[str]:
        """Detect RSI divergence (simplified)."""
        # TODO: Implement proper divergence detection
        return None

    def _detect_macd_divergence(
        self, prices: pd.Series, current_histogram: float
    ) -> Optional[str]:
        """Detect MACD divergence (simplified)."""
        # TODO: Implement proper divergence detection
        return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_indicator_engine.py::TestIndicatorEngineCalculateAll -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/indicator_engine.py tests/validators/test_indicator_engine.py
git commit -m "feat(validators): add calculate_all method to indicator engine"
```

---

## Task 6: Veto Logic

**Files:**
- Create: `src/validators/veto_logic.py`
- Test: `tests/validators/test_veto_logic.py`

**Step 1: Write the failing test**

```python
# tests/validators/test_veto_logic.py
import pytest
from src.validators.veto_logic import VetoLogic
from src.validators.models import TechnicalIndicators, ValidationStatus
from src.analyzers.sentiment_result import SentimentLabel


class TestVetoLogicRSI:
    @pytest.fixture
    def logic(self):
        return VetoLogic(
            rsi_overbought=70,
            rsi_oversold=30,
            adx_trend_threshold=20,
        )

    def test_bullish_signal_vetoed_when_overbought(self, logic):
        indicators = TechnicalIndicators(
            rsi=75.0,
            macd_histogram=0.1,
            macd_trend="rising",
            stochastic_k=80.0,
            stochastic_d=75.0,
            adx=25.0,
        )
        status, reasons = logic.evaluate(indicators, SentimentLabel.BULLISH)
        assert status == ValidationStatus.VETO
        assert any("overbought" in r.lower() for r in reasons)

    def test_bearish_signal_vetoed_when_oversold(self, logic):
        indicators = TechnicalIndicators(
            rsi=25.0,
            macd_histogram=-0.1,
            macd_trend="falling",
            stochastic_k=20.0,
            stochastic_d=25.0,
            adx=25.0,
        )
        status, reasons = logic.evaluate(indicators, SentimentLabel.BEARISH)
        assert status == ValidationStatus.VETO
        assert any("oversold" in r.lower() for r in reasons)

    def test_bullish_signal_passes_normal_rsi(self, logic):
        indicators = TechnicalIndicators(
            rsi=55.0,
            macd_histogram=0.1,
            macd_trend="rising",
            stochastic_k=60.0,
            stochastic_d=55.0,
            adx=25.0,
        )
        status, reasons = logic.evaluate(indicators, SentimentLabel.BULLISH)
        assert status == ValidationStatus.PASS
        assert len(reasons) == 0


class TestVetoLogicADX:
    @pytest.fixture
    def logic(self):
        return VetoLogic(
            rsi_overbought=70,
            rsi_oversold=30,
            adx_trend_threshold=20,
        )

    def test_signal_vetoed_when_no_trend(self, logic):
        indicators = TechnicalIndicators(
            rsi=50.0,
            macd_histogram=0.0,
            macd_trend="flat",
            stochastic_k=50.0,
            stochastic_d=50.0,
            adx=15.0,  # Below threshold
        )
        status, reasons = logic.evaluate(indicators, SentimentLabel.BULLISH)
        assert status == ValidationStatus.VETO
        assert any("adx" in r.lower() or "trend" in r.lower() for r in reasons)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_veto_logic.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/validators/veto_logic.py
"""Veto logic for technical validation."""
from src.validators.models import TechnicalIndicators, ValidationStatus
from src.analyzers.sentiment_result import SentimentLabel


class VetoLogic:
    """Applies veto rules to technical indicators."""

    def __init__(
        self,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        adx_trend_threshold: float = 20.0,
    ):
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.adx_trend_threshold = adx_trend_threshold

    def evaluate(
        self,
        indicators: TechnicalIndicators,
        sentiment: SentimentLabel,
    ) -> tuple[ValidationStatus, list[str]]:
        """
        Evaluate technical indicators against veto rules.

        Args:
            indicators: Technical indicator values
            sentiment: Sentiment direction from Phase 2

        Returns:
            Tuple of (status, list of veto reasons)
        """
        reasons = []

        # RSI checks
        if sentiment == SentimentLabel.BULLISH:
            if indicators.rsi > self.rsi_overbought:
                reasons.append(f"RSI overbought at {indicators.rsi:.1f}")
        elif sentiment == SentimentLabel.BEARISH:
            if indicators.rsi < self.rsi_oversold:
                reasons.append(f"RSI oversold at {indicators.rsi:.1f}")

        # ADX trend check
        if indicators.adx < self.adx_trend_threshold:
            reasons.append(
                f"ADX {indicators.adx:.1f} below trend threshold {self.adx_trend_threshold}"
            )

        # MACD divergence check
        if sentiment == SentimentLabel.BULLISH and indicators.macd_divergence == "bearish":
            reasons.append("Bearish MACD divergence detected")
        elif sentiment == SentimentLabel.BEARISH and indicators.macd_divergence == "bullish":
            reasons.append("Bullish MACD divergence detected")

        # MACD momentum fading check
        if sentiment == SentimentLabel.BULLISH and indicators.macd_trend == "falling":
            if indicators.macd_histogram < 0:
                reasons.append("MACD momentum fading (histogram negative and falling)")
        elif sentiment == SentimentLabel.BEARISH and indicators.macd_trend == "rising":
            if indicators.macd_histogram > 0:
                reasons.append("MACD momentum strengthening against bearish signal")

        if reasons:
            return ValidationStatus.VETO, reasons

        return ValidationStatus.PASS, []
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_veto_logic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/veto_logic.py tests/validators/test_veto_logic.py
git commit -m "feat(validators): add veto logic for technical validation"
```

---

## Task 7: Options Validator

**Files:**
- Create: `src/validators/options_validator.py`
- Test: `tests/validators/test_options_validator.py`

**Step 1: Write the failing test**

```python
# tests/validators/test_options_validator.py
import pytest
from src.validators.options_validator import OptionsValidator
from src.validators.models import OptionsFlowData


class TestOptionsValidator:
    @pytest.fixture
    def validator(self):
        return OptionsValidator(
            volume_spike_ratio=2.0,
            iv_rank_warning_threshold=80.0,
        )

    def test_detect_volume_spike(self, validator):
        data = OptionsFlowData(
            volume_ratio=3.5,
            iv_rank=45.0,
            put_call_ratio=0.8,
            unusual_activity=True,
        )
        is_enhanced, warnings = validator.validate(data)
        assert is_enhanced is True
        assert len(warnings) == 0

    def test_high_iv_warning(self, validator):
        data = OptionsFlowData(
            volume_ratio=1.5,
            iv_rank=85.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        is_enhanced, warnings = validator.validate(data)
        assert is_enhanced is False
        assert any("IV" in w or "volatility" in w.lower() for w in warnings)

    def test_normal_options_flow(self, validator):
        data = OptionsFlowData(
            volume_ratio=1.2,
            iv_rank=40.0,
            put_call_ratio=0.9,
            unusual_activity=False,
        )
        is_enhanced, warnings = validator.validate(data)
        assert is_enhanced is False
        assert len(warnings) == 0

    def test_validate_returns_modifier(self, validator):
        data = OptionsFlowData(
            volume_ratio=3.0,
            iv_rank=30.0,
            put_call_ratio=0.5,
            unusual_activity=True,
        )
        is_enhanced, warnings = validator.validate(data)
        modifier = validator.get_confidence_modifier(data)
        assert modifier > 1.0  # Enhanced signal
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_options_validator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/validators/options_validator.py
"""Options flow validation."""
from src.validators.models import OptionsFlowData


class OptionsValidator:
    """Validates options flow data."""

    def __init__(
        self,
        volume_spike_ratio: float = 2.0,
        iv_rank_warning_threshold: float = 80.0,
    ):
        self.volume_spike_ratio = volume_spike_ratio
        self.iv_rank_warning_threshold = iv_rank_warning_threshold

    def validate(self, data: OptionsFlowData) -> tuple[bool, list[str]]:
        """
        Validate options flow data.

        Args:
            data: Options flow data

        Returns:
            Tuple of (is_enhanced, list of warnings)
        """
        warnings = []
        is_enhanced = False

        # Volume spike detection
        if data.volume_ratio >= self.volume_spike_ratio:
            is_enhanced = True

        # High IV warning
        if data.iv_rank > self.iv_rank_warning_threshold:
            warnings.append(
                f"High IV rank ({data.iv_rank:.1f}%) - options expensive, possible event"
            )

        # Unusual activity flag
        if data.unusual_activity and data.volume_ratio >= self.volume_spike_ratio:
            is_enhanced = True

        return is_enhanced, warnings

    def get_confidence_modifier(self, data: OptionsFlowData) -> float:
        """
        Calculate confidence modifier based on options flow.

        Returns:
            Modifier between 0.8 and 1.3
        """
        modifier = 1.0

        # Volume spike boosts confidence
        if data.volume_ratio >= self.volume_spike_ratio:
            modifier += 0.1

        # Unusual activity boosts confidence
        if data.unusual_activity:
            modifier += 0.1

        # Low IV is favorable
        if data.iv_rank < 50.0:
            modifier += 0.1

        # High IV reduces confidence
        if data.iv_rank > self.iv_rank_warning_threshold:
            modifier -= 0.2

        return max(0.8, min(1.3, modifier))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_options_validator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/options_validator.py tests/validators/test_options_validator.py
git commit -m "feat(validators): add options flow validator"
```

---

## Task 8: Technical Validator Settings

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Test: `tests/config/test_validator_settings.py`

**Step 1: Write the failing test**

```python
# tests/config/test_validator_settings.py
import pytest
from pydantic import ValidationError
from src.config.settings import Settings, TechnicalValidatorSettings


class TestTechnicalValidatorSettings:
    def test_settings_has_validators_config(self):
        settings = Settings()
        assert hasattr(settings, "validators")
        assert hasattr(settings.validators, "technical")

    def test_default_values(self):
        settings = Settings()
        tech = settings.validators.technical
        assert tech.enabled is True
        assert tech.rsi_period == 14
        assert tech.rsi_overbought == 70.0
        assert tech.rsi_oversold == 30.0
        assert tech.adx_trend_threshold == 20.0
        assert tech.veto_mode is True

    def test_rsi_overbought_validation(self):
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(rsi_overbought=40.0)  # Below min 50

    def test_rsi_oversold_validation(self):
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(rsi_oversold=60.0)  # Above max 50

    def test_lookback_bars_validation(self):
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(lookback_bars=10)  # Below min 20
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_validator_settings.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Add to src/config/settings.py

class TechnicalValidatorSettings(BaseModel):
    """Settings for technical validator."""

    enabled: bool = True
    rsi_period: int = Field(default=14, ge=2, le=50)
    rsi_overbought: float = Field(default=70.0, ge=50.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=50.0)
    macd_fast: int = Field(default=12, ge=5, le=20)
    macd_slow: int = Field(default=26, ge=20, le=50)
    macd_signal: int = Field(default=9, ge=5, le=15)
    stoch_k_period: int = Field(default=14, ge=5, le=30)
    stoch_d_period: int = Field(default=3, ge=2, le=10)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_trend_threshold: float = Field(default=20.0, ge=10.0, le=40.0)
    options_volume_spike_ratio: float = Field(default=2.0, ge=1.0, le=10.0)
    iv_rank_warning_threshold: float = Field(default=80.0, ge=50.0, le=100.0)
    veto_mode: bool = True
    lookback_bars: int = Field(default=50, ge=20, le=200)
    timeframe: str = "5Min"


class ValidatorsSettings(BaseModel):
    """Settings for all validators."""

    technical: TechnicalValidatorSettings = Field(
        default_factory=TechnicalValidatorSettings
    )


# Modify Settings class to add validators
class Settings(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    analyzers: AnalyzersSettings = Field(default_factory=AnalyzersSettings)
    validators: ValidatorsSettings = Field(default_factory=ValidatorsSettings)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    reddit_api: RedditAPIConfig = Field(default_factory=RedditAPIConfig)
```

```yaml
# Add to config/settings.yaml

# CAPA 3: Technical Validation
validators:
  technical:
    enabled: true
    rsi_period: 14
    rsi_overbought: 70.0
    rsi_oversold: 30.0
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    stoch_k_period: 14
    stoch_d_period: 3
    adx_period: 14
    adx_trend_threshold: 20.0
    options_volume_spike_ratio: 2.0
    iv_rank_warning_threshold: 80.0
    veto_mode: true
    lookback_bars: 50
    timeframe: "5Min"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_validator_settings.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config/settings.py config/settings.yaml tests/config/test_validator_settings.py
git commit -m "feat(config): add technical validator settings"
```

---

## Task 9: Alpaca Market Data Client Extension

**Files:**
- Modify: `src/clients/alpaca_client.py`
- Test: `tests/clients/test_alpaca_market_data.py`

**Step 1: Write the failing test**

```python
# tests/clients/test_alpaca_market_data.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.clients.alpaca_client import AlpacaClient


class TestAlpacaMarketData:
    @pytest.fixture
    def mock_client(self):
        with patch("src.clients.alpaca_client.StockHistoricalDataClient") as mock:
            client = AlpacaClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True,
            )
            client._data_client = mock.return_value
            yield client

    def test_get_bars_returns_dataframe(self, mock_client):
        # Mock the response
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        })
        mock_client._data_client.get_stock_bars.return_value = mock_bars

        result = mock_client.get_bars("NVDA", limit=50, timeframe="5Min")

        assert isinstance(result, pd.DataFrame)
        assert "open" in result.columns
        assert "close" in result.columns

    def test_get_bars_with_symbol(self, mock_client):
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame({
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000],
        })
        mock_client._data_client.get_stock_bars.return_value = mock_bars

        mock_client.get_bars("AAPL", limit=30, timeframe="1Min")

        mock_client._data_client.get_stock_bars.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/clients/test_alpaca_market_data.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Add to src/clients/alpaca_client.py

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from datetime import datetime, timezone, timedelta


class AlpacaClient:
    # ... existing code ...

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        # ... existing init ...
        self._data_client = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(
        self,
        symbol: str,
        limit: int = 50,
        timeframe: str = "5Min",
    ) -> pd.DataFrame:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock symbol
            limit: Number of bars to fetch
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")

        Returns:
            DataFrame with OHLCV data
        """
        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }

        tf = tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            limit=limit,
        )

        bars = self._data_client.get_stock_bars(request)

        if bars.df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        return df
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/clients/test_alpaca_market_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/clients/alpaca_client.py tests/clients/test_alpaca_market_data.py
git commit -m "feat(clients): add market data methods to AlpacaClient"
```

---

## Task 10: Technical Validator Orchestrator

**Files:**
- Create: `src/validators/technical_validator.py`
- Test: `tests/validators/test_technical_validator.py`

**Step 1: Write the failing test**

```python
# tests/validators/test_technical_validator.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timezone

from src.validators.technical_validator import TechnicalValidator
from src.validators.models import ValidationStatus, TechnicalIndicators
from src.analyzers.analyzer_manager import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.models.social_message import SocialMessage, SourceType


class TestTechnicalValidator:
    @pytest.fixture
    def mock_alpaca_client(self):
        mock = MagicMock()
        mock.get_bars.return_value = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [1000] * 60,
        })
        return mock

    @pytest.fixture
    def sample_analyzed_message(self):
        message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test_user",
            content="$NVDA looking bullish!",
            timestamp=datetime.now(timezone.utc),
        )
        sentiment = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.85,
            confidence=0.80,
        )
        return AnalyzedMessage(
            message=message,
            sentiment=sentiment,
            deep_analysis=None,
        )

    def test_validate_returns_validated_signal(
        self, mock_alpaca_client, sample_analyzed_message
    ):
        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
        )

        with patch.object(
            validator._indicator_engine,
            "calculate_all",
            return_value=TechnicalIndicators(
                rsi=55.0,
                macd_histogram=0.1,
                macd_trend="rising",
                stochastic_k=60.0,
                stochastic_d=55.0,
                adx=25.0,
            ),
        ):
            result = validator.validate(sample_analyzed_message, symbol="NVDA")

        assert result.validation.status == ValidationStatus.PASS
        assert result.should_trade() is True

    def test_validate_vetos_overbought(
        self, mock_alpaca_client, sample_analyzed_message
    ):
        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            rsi_overbought=70.0,
        )

        with patch.object(
            validator._indicator_engine,
            "calculate_all",
            return_value=TechnicalIndicators(
                rsi=75.0,  # Overbought
                macd_histogram=0.1,
                macd_trend="rising",
                stochastic_k=80.0,
                stochastic_d=75.0,
                adx=25.0,
            ),
        ):
            result = validator.validate(sample_analyzed_message, symbol="NVDA")

        assert result.validation.status == ValidationStatus.VETO
        assert result.should_trade() is False
        assert len(result.validation.veto_reasons) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_technical_validator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/validators/technical_validator.py
"""Technical validator orchestrator."""
import logging
from typing import Optional

import pandas as pd

from src.analyzers.analyzer_manager import AnalyzedMessage
from src.validators.indicator_engine import IndicatorEngine
from src.validators.veto_logic import VetoLogic
from src.validators.options_validator import OptionsValidator
from src.validators.models import (
    TechnicalIndicators,
    TechnicalValidation,
    ValidationStatus,
    ValidatedSignal,
    OptionsFlowData,
)

logger = logging.getLogger(__name__)


class TechnicalValidator:
    """Orchestrates technical validation of trading signals."""

    def __init__(
        self,
        alpaca_client,
        veto_mode: bool = True,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        adx_trend_threshold: float = 20.0,
        lookback_bars: int = 50,
        timeframe: str = "5Min",
        options_volume_spike_ratio: float = 2.0,
        iv_rank_warning_threshold: float = 80.0,
    ):
        self._alpaca_client = alpaca_client
        self._veto_mode = veto_mode
        self._lookback_bars = lookback_bars
        self._timeframe = timeframe

        self._indicator_engine = IndicatorEngine()
        self._veto_logic = VetoLogic(
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            adx_trend_threshold=adx_trend_threshold,
        )
        self._options_validator = OptionsValidator(
            volume_spike_ratio=options_volume_spike_ratio,
            iv_rank_warning_threshold=iv_rank_warning_threshold,
        )

    def validate(
        self,
        analyzed_message: AnalyzedMessage,
        symbol: str,
        options_data: Optional[OptionsFlowData] = None,
    ) -> ValidatedSignal:
        """
        Validate an analyzed message with technical indicators.

        Args:
            analyzed_message: Message with sentiment analysis from Phase 2
            symbol: Stock symbol to fetch data for
            options_data: Optional options flow data

        Returns:
            ValidatedSignal with technical validation result
        """
        try:
            # Fetch market data
            ohlc = self._alpaca_client.get_bars(
                symbol=symbol,
                limit=self._lookback_bars,
                timeframe=self._timeframe,
            )

            if ohlc.empty:
                logger.warning(f"No market data for {symbol}, passing signal")
                return self._create_pass_signal(analyzed_message, options_data)

            # Calculate indicators
            indicators = self._indicator_engine.calculate_all(ohlc)

            # Apply veto logic
            sentiment_label = analyzed_message.sentiment.label
            status, veto_reasons = self._veto_logic.evaluate(indicators, sentiment_label)

            # Process options data if available
            warnings = []
            confidence_modifier = 1.0

            if options_data:
                is_enhanced, opt_warnings = self._options_validator.validate(options_data)
                warnings.extend(opt_warnings)
                confidence_modifier = self._options_validator.get_confidence_modifier(
                    options_data
                )

            # In warn mode, convert VETO to WARN
            if not self._veto_mode and status == ValidationStatus.VETO:
                status = ValidationStatus.WARN
                warnings.extend(veto_reasons)
                veto_reasons = []

            validation = TechnicalValidation(
                status=status,
                indicators=indicators,
                options_flow=options_data,
                veto_reasons=veto_reasons,
                warnings=warnings,
                confidence_modifier=confidence_modifier,
            )

            return ValidatedSignal(
                message=analyzed_message,
                validation=validation,
            )

        except Exception as e:
            logger.error(f"Technical validation failed: {e}")
            return self._create_pass_signal(analyzed_message, options_data)

    def _create_pass_signal(
        self,
        analyzed_message: AnalyzedMessage,
        options_data: Optional[OptionsFlowData],
    ) -> ValidatedSignal:
        """Create a passing signal for error cases."""
        indicators = TechnicalIndicators(
            rsi=50.0,
            macd_histogram=0.0,
            macd_trend="flat",
            stochastic_k=50.0,
            stochastic_d=50.0,
            adx=0.0,
        )
        validation = TechnicalValidation(
            status=ValidationStatus.PASS,
            indicators=indicators,
            options_flow=options_data,
            veto_reasons=[],
            warnings=["Technical validation skipped due to missing data"],
            confidence_modifier=0.8,
        )
        return ValidatedSignal(message=analyzed_message, validation=validation)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_technical_validator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/validators/technical_validator.py tests/validators/test_technical_validator.py
git commit -m "feat(validators): add TechnicalValidator orchestrator"
```

---

## Task 11: Update Module Exports

**Files:**
- Modify: `src/validators/__init__.py`

**Step 1: Update exports**

```python
# src/validators/__init__.py
"""Technical validation module for CAPA 3."""

from .models import (
    TechnicalIndicators,
    OptionsFlowData,
    ValidationStatus,
    TechnicalValidation,
    ValidatedSignal,
)
from .indicator_engine import IndicatorEngine
from .veto_logic import VetoLogic
from .options_validator import OptionsValidator
from .technical_validator import TechnicalValidator

__all__ = [
    "TechnicalIndicators",
    "OptionsFlowData",
    "ValidationStatus",
    "TechnicalValidation",
    "ValidatedSignal",
    "IndicatorEngine",
    "VetoLogic",
    "OptionsValidator",
    "TechnicalValidator",
]
```

**Step 2: Commit**

```bash
git add src/validators/__init__.py
git commit -m "chore(validators): update module exports"
```

---

## Task 12: Integration Tests

**Files:**
- Create: `tests/integration/test_validation_pipeline.py`

**Step 1: Write integration tests**

```python
# tests/integration/test_validation_pipeline.py
import pytest
from unittest.mock import MagicMock
import pandas as pd
from datetime import datetime, timezone

from src.validators.technical_validator import TechnicalValidator
from src.validators.models import ValidationStatus, OptionsFlowData
from src.analyzers.analyzer_manager import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.models.social_message import SocialMessage, SourceType


class TestValidationPipelineIntegration:
    @pytest.fixture
    def mock_alpaca_client(self):
        mock = MagicMock()
        return mock

    @pytest.fixture
    def bullish_message(self):
        message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="$NVDA large call sweep detected",
            timestamp=datetime.now(timezone.utc),
        )
        sentiment = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.90,
            confidence=0.85,
        )
        return AnalyzedMessage(
            message=message,
            sentiment=sentiment,
            deep_analysis=None,
        )

    def test_full_pipeline_pass(self, mock_alpaca_client, bullish_message):
        """Test full pipeline: bullish signal with favorable technicals → PASS."""
        # Setup favorable market data (uptrend)
        mock_alpaca_client.get_bars.return_value = pd.DataFrame({
            "open": [100.0 + i * 0.5 for i in range(60)],
            "high": [101.0 + i * 0.5 for i in range(60)],
            "low": [99.0 + i * 0.5 for i in range(60)],
            "close": [100.5 + i * 0.5 for i in range(60)],
            "volume": [1000] * 60,
        })

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
        )

        result = validator.validate(bullish_message, symbol="NVDA")

        assert result.should_trade() is True
        assert result.validation.indicators.rsi > 0

    def test_full_pipeline_veto_overbought(self, mock_alpaca_client, bullish_message):
        """Test full pipeline: bullish signal with overbought RSI → VETO."""
        # Setup overbought market data (strong uptrend)
        prices = [100.0]
        for _ in range(59):
            prices.append(prices[-1] * 1.02)

        mock_alpaca_client.get_bars.return_value = pd.DataFrame({
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1000] * 60,
        })

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            rsi_overbought=70.0,
        )

        result = validator.validate(bullish_message, symbol="NVDA")

        # Should veto due to overbought conditions
        assert result.validation.indicators.rsi > 70
        assert result.validation.status == ValidationStatus.VETO

    def test_pipeline_with_options_data(self, mock_alpaca_client, bullish_message):
        """Test pipeline with options flow data enhancement."""
        mock_alpaca_client.get_bars.return_value = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [1000] * 60,
        })

        options_data = OptionsFlowData(
            volume_ratio=3.5,
            iv_rank=35.0,
            put_call_ratio=0.6,
            unusual_activity=True,
        )

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
        )

        result = validator.validate(
            bullish_message,
            symbol="NVDA",
            options_data=options_data,
        )

        assert result.validation.options_flow is not None
        assert result.validation.confidence_modifier > 1.0  # Enhanced

    def test_warn_mode_converts_veto(self, mock_alpaca_client, bullish_message):
        """Test warn mode converts VETO to WARN."""
        # Overbought data
        prices = [100.0]
        for _ in range(59):
            prices.append(prices[-1] * 1.02)

        mock_alpaca_client.get_bars.return_value = pd.DataFrame({
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1000] * 60,
        })

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=False,  # Warn mode
        )

        result = validator.validate(bullish_message, symbol="NVDA")

        # Should warn instead of veto
        assert result.validation.status == ValidationStatus.WARN
        assert result.should_trade() is True
        assert len(result.validation.warnings) > 0
```

**Step 2: Run integration tests**

Run: `pytest tests/integration/test_validation_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_validation_pipeline.py
git commit -m "test: add integration tests for validation pipeline"
```

---

## Task 13: Final Test Suite Run

**Step 1: Run all tests**

Run: `pytest --cov=src --cov-report=term-missing -v`
Expected: All tests pass with >90% coverage on new validators module

**Step 2: Commit test coverage baseline**

```bash
git add .
git commit -m "test: complete Phase 3 test suite"
```

---

## Summary

Phase 3 adds technical validation (CAPA 3) with:

1. **Data Models** - TechnicalIndicators, OptionsFlowData, ValidatedSignal
2. **Indicator Engine** - RSI, MACD, Stochastic, ADX calculations using pandas-ta
3. **Veto Logic** - Rules to block dangerous entries (overbought/oversold, no trend)
4. **Options Validator** - Volume spike and IV rank validation
5. **Technical Validator** - Orchestrator combining all components
6. **Settings** - Configurable thresholds for all indicators
7. **Integration Tests** - Full pipeline testing

**Dependencies to add:**
```bash
pip install pandas-ta
```
