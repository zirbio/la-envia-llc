# TechnicalValidator Usage Guide

## Overview

The `TechnicalValidator` orchestrates the technical validation pipeline for trading signals. It combines:
- **IndicatorEngine**: Calculates technical indicators (RSI, MACD, Stochastic, ADX)
- **VetoLogic**: Applies veto rules based on indicator conflicts
- **OptionsValidator**: Processes options flow data for additional signal quality

## Quick Start

```python
from src.validators import TechnicalValidator
from src.analyzers import AnalyzedMessage
from alpaca.data.historical import StockHistoricalDataClient

# Initialize Alpaca client
alpaca_client = StockHistoricalDataClient(api_key="...", secret_key="...")

# Create validator
validator = TechnicalValidator(
    alpaca_client=alpaca_client,
    veto_mode=True,  # Enable veto logic
    rsi_overbought=70.0,
    rsi_oversold=30.0,
    adx_trend_threshold=20.0,
    lookback_bars=50,
    timeframe="5Min",
)

# Validate a signal
validated_signal = validator.validate(
    analyzed_message=analyzed_message,
    symbol="AAPL",
)

# Check if signal should be traded
if validated_signal.should_trade():
    print("Signal is valid for trading!")
    print(f"Confidence: {validated_signal.validation.confidence_modifier}")
else:
    print("Signal vetoed:")
    for reason in validated_signal.validation.veto_reasons:
        print(f"  - {reason}")
```

## Configuration Parameters

### Core Settings
- `alpaca_client`: Market data client (duck-typed, must implement `get_bars()`)
- `veto_mode`: If `True`, veto signals; if `False`, convert to warnings
- `lookback_bars`: Number of historical bars to fetch (default: 50)
- `timeframe`: Bar timeframe (default: "5Min")

### Indicator Thresholds
- `rsi_overbought`: RSI threshold for overbought (default: 70.0)
- `rsi_oversold`: RSI threshold for oversold (default: 30.0)
- `adx_trend_threshold`: Minimum ADX for trend strength (default: 20.0)

### Options Settings
- `options_volume_spike_ratio`: Volume spike threshold (default: 2.0)
- `iv_rank_warning_threshold`: IV rank warning level (default: 80.0)

## Validation Pipeline

### Step 1: Fetch Market Data
```python
# Validator calls: alpaca_client.get_bars(symbol, timeframe, limit)
ohlc_data = validator._fetch_market_data("AAPL")
```

### Step 2: Calculate Indicators
```python
# IndicatorEngine calculates all technical indicators
indicators = validator.indicator_engine.calculate_all(ohlc_data)
# Returns: TechnicalIndicators with RSI, MACD, Stochastic, ADX
```

### Step 3: Apply Veto Logic
```python
# VetoLogic evaluates indicators against sentiment
status, veto_reasons = validator.veto_logic.evaluate(
    indicators,
    analyzed_message.sentiment.label
)
```

### Step 4: Process Options Data (Optional)
```python
from src.validators import OptionsFlowData

options_data = OptionsFlowData(
    volume_ratio=3.0,  # 300% of average
    iv_rank=50.0,
    put_call_ratio=0.8,
    unusual_activity=True,
)

validated_signal = validator.validate(
    analyzed_message=analyzed_message,
    symbol="AAPL",
    options_data=options_data,  # Optional
)

# Options data enhances confidence modifier
print(validated_signal.validation.confidence_modifier)  # > 1.0
```

## Validation Statuses

### PASS
Signal passes all validation checks and can be traded.

```python
if validated_signal.validation.status == ValidationStatus.PASS:
    # Safe to trade
    pass
```

### VETO
Signal is blocked due to technical indicator conflicts.

```python
if validated_signal.validation.status == ValidationStatus.VETO:
    print("Veto reasons:", validated_signal.validation.veto_reasons)
    # Do not trade
```

### WARN
Signal has warnings but is not blocked (only when `veto_mode=False`).

```python
if validated_signal.validation.status == ValidationStatus.WARN:
    print("Warnings:", validated_signal.validation.warnings)
    # Trade with caution
```

## Veto Rules

### Rule 1: Overbought Bullish
Bullish signals are vetoed when RSI > `rsi_overbought` threshold.

### Rule 2: Oversold Bearish
Bearish signals are vetoed when RSI < `rsi_oversold` threshold.

### Rule 3: Weak Trend
All signals are vetoed when ADX < `adx_trend_threshold`.

### Rule 4-7: MACD Divergence
- Bullish vetoed if MACD divergence is bearish
- Bearish vetoed if MACD divergence is bullish
- Bullish vetoed if MACD falling with negative histogram
- Bearish vetoed if MACD rising with positive histogram

## Error Handling

The validator handles errors gracefully with fail-safe behavior:

```python
try:
    validated_signal = validator.validate(
        analyzed_message=analyzed_message,
        symbol="AAPL",
    )
except Exception:
    # Never raises - returns PASS with warning instead
    pass

# On API error:
assert validated_signal.validation.status == ValidationStatus.PASS
assert len(validated_signal.validation.warnings) > 0
assert "failed" in validated_signal.validation.warnings[0].lower()
```

## Example: Full Workflow

```python
from src.collectors import StocktwitsCollector
from src.analyzers import AnalyzerManager, SentimentAnalyzer
from src.validators import TechnicalValidator
from alpaca.data.historical import StockHistoricalDataClient

# 1. Setup components
collector = StocktwitsCollector(watchlist=["AAPL", "TSLA"])
sentiment_analyzer = SentimentAnalyzer()
analyzer_manager = AnalyzerManager(sentiment_analyzer=sentiment_analyzer)
alpaca_client = StockHistoricalDataClient(api_key="...", secret_key="...")
validator = TechnicalValidator(alpaca_client=alpaca_client)

# 2. Collect and analyze messages
collector.connect()
for social_message in collector.stream():
    analyzed_message = analyzer_manager.analyze(social_message)

    # 3. Extract tickers and validate
    for ticker in analyzed_message.get_tickers():
        validated_signal = validator.validate(
            analyzed_message=analyzed_message,
            symbol=ticker,
        )

        # 4. Trade if valid
        if validated_signal.should_trade():
            confidence = validated_signal.validation.confidence_modifier
            print(f"Trading {ticker} with confidence {confidence:.2f}")
        else:
            print(f"Skipping {ticker}: {validated_signal.validation.veto_reasons}")
```

## Testing

The validator includes comprehensive test coverage with mocked dependencies:

```bash
# Run validator tests
uv run pytest tests/validators/test_technical_validator.py -v

# Run all validator tests
uv run pytest tests/validators/ -v
```

## Related Documentation

- [Indicator Engine](./indicator_engine.md)
- [Veto Logic](./veto_logic.md)
- [Options Validator](./options_validator.md)
- [Phase 3 Architecture](./phase3_architecture.md)
