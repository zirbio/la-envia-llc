# Main.py Integration Design

**Date:** 2026-01-18
**Status:** Approved
**Purpose:** Integrate all 7-layer pipeline components into main.py entry point

---

## Overview

This design integrates the complete trading system pipeline into `main.py`, coordinating:
- Layer 0: Market Gate
- Layer 1: Collectors (Stocktwits, Alpaca News, Reddit)
- Layer 2: Analyzers (FinTwitBERT + Claude)
- Layer 3: Technical Validation
- Layer 4: Scoring System
- Layer 5: Risk Management
- Layer 6: Trade Execution
- Observability: Notifications (Telegram) + Journal

The `TradingOrchestrator` serves as the central coordinator, receiving messages from collectors and routing them through the pipeline automatically.

---

## High-Level Architecture

### Data Flow

```
Collectors (Layer 1) → Analyzers (Layer 2) → Validators (Layer 3)
→ Scorer (Layer 4) → Risk Manager (Layer 5) → Market Gate (Layer 0)
→ Executor (Layer 6) → Journal & Notifications (observability)
```

### Key Design Principles

1. **Single Settings Hub**: One `Settings` object loaded from `config/settings.yaml` provides configuration to all components
2. **Shared AlpacaClient**: Initialized once, shared via dependency injection to Market Gate, Validators, and Executor
3. **TradingOrchestrator**: Central coordinator that receives messages from collectors and routes them through pipeline
4. **Passive Observers**: Notification & Journal are passive observers that TradeExecutor calls on trade events
5. **Fail Fast**: Any critical component failure stops the system immediately with clear error messages

### Component Relationships

- `CollectorManager` manages Stocktwits/AlpacaNews/Reddit collectors
- `AnalyzerManager` chains FinTwitBERT → Claude analysis
- `SignalScorer` composes 5 sub-components (credibility, time factors, confluence, weights, builder)
- `TradeExecutor` coordinates RiskManager + JournalManager + TelegramNotifier

---

## Initialization Sequence

Sequential initialization for fail-fast debugging:

### Phase 1: Configuration
```python
1. Load .env variables
2. Load Settings from config/settings.yaml
3. Validate critical env vars (ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY)
4. Create data directories if missing (data/trades, data/signals, data/cache)
```

**Failure behavior:** Stop immediately, show which config is missing

### Phase 2: Core Infrastructure
```python
5. Initialize AlpacaClient
6. Connect to Alpaca and verify account
7. Initialize TelegramNotifier
8. Test Telegram connection (send startup message)
```

**Failure behavior:** Stop if Alpaca fails, stop if Telegram fails

### Phase 3: Analysis Components
```python
9. Initialize SentimentAnalyzer (download FinTwitBERT if needed)
10. Initialize ClaudeAnalyzer (test API connection)
11. Initialize AnalyzerManager (combines both)
```

**Failure behavior:** Stop if FinTwitBERT download fails, stop if Claude API fails

### Phase 4: Collectors
```python
12. Initialize StocktwitsCollector (no auth required)
13. Initialize AlpacaNewsCollector (uses existing AlpacaClient)
14. Initialize RedditCollector (skip if REDDIT_CLIENT_ID missing)
15. Create CollectorManager with available collectors
```

**Failure behavior:** Warn if Reddit skipped, stop if no collectors available

### Phase 5: Pipeline Components
```python
16. Initialize TechnicalValidator (needs AlpacaClient)
17. Initialize SignalScorer + all 5 sub-components
18. Initialize MarketGate (needs AlpacaClient)
19. Initialize RiskManager
20. Initialize JournalManager
21. Initialize TradeExecutor (needs AlpacaClient + RiskManager)
```

**Failure behavior:** Stop if any component fails

### Phase 6: Orchestrator
```python
22. Create TradingOrchestrator with all components
23. Start orchestrator (begins streaming messages)
```

**Failure behavior:** Stop if orchestrator fails to start

---

## Error Handling & Validation

### Configuration Validation

```
Missing .env file:
→ "ERROR: .env file not found. Run: cp .env.example .env"

Missing YAML:
→ "ERROR: config/settings.yaml not found"

Missing critical env vars:
→ "ERROR: ALPACA_API_KEY not set in .env"
→ "ERROR: ANTHROPIC_API_KEY not set in .env"
```

### Connection Validation

```
Alpaca connection failed:
→ "ERROR: Failed to connect to Alpaca: [reason]"
→ "Check API keys in .env and try again"

Alpaca account check success:
→ "Account cash: $X,XXX.XX"

Telegram failed:
→ "ERROR: Telegram connection failed: [reason]"
→ "Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
```

### Model/API Validation

```
FinTwitBERT download failed:
→ "ERROR: Failed to download FinTwitBERT model"
→ "Check internet connection"

Claude API failed:
→ "ERROR: Claude API test failed: [reason]"
→ "Check ANTHROPIC_API_KEY in .env"
```

### Collector Validation

```
Reddit credentials missing (warning, not error):
→ "WARNING: Reddit credentials not found - skipping Reddit collector"
→ "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to enable"

No collectors available:
→ "ERROR: No collectors available. At least Stocktwits should work."
```

### Success Output

```
✓ Settings loaded from config/settings.yaml
✓ Alpaca connected (Paper mode: True, Cash: $100,000.00)
✓ Telegram connected (sent startup message)
✓ FinTwitBERT model loaded
✓ Claude API verified
✓ Collectors: Stocktwits, Alpaca News, Reddit (3/3)
✓ TradingOrchestrator started

Listening for signals...
```

---

## Component Wiring & Dependencies

### SignalScorer Assembly

SignalScorer requires 5 sub-components, all configured from `settings.scoring`:

```python
# 1. Source Credibility Manager
credibility_manager = SourceCredibilityManager(
    tier1_sources=settings.scoring.tier1_sources,
    tier1_multiplier=settings.scoring.credibility_tier1_multiplier,
    tier2_multiplier=settings.scoring.credibility_tier2_multiplier,
    tier3_multiplier=settings.scoring.credibility_tier3_multiplier,
)

# 2. Time Factor Calculator
time_calculator = TimeFactorCalculator(
    timezone=settings.system.timezone,
    premarket_factor=settings.scoring.premarket_factor,
    afterhours_factor=settings.scoring.afterhours_factor,
    earnings_factor=settings.scoring.earnings_factor,
    earnings_proximity_days=settings.scoring.earnings_proximity_days,
)

# 3. Confluence Detector
confluence_detector = ConfluenceDetector(
    window_minutes=settings.scoring.confluence_window_minutes,
    bonus_2_signals=settings.scoring.confluence_bonus_2_signals,
    bonus_3_signals=settings.scoring.confluence_bonus_3_signals,
)

# 4. Dynamic Weight Calculator
weight_calculator = DynamicWeightCalculator(
    base_sentiment_weight=settings.scoring.base_sentiment_weight,
    base_technical_weight=settings.scoring.base_technical_weight,
    strong_trend_adx=settings.scoring.strong_trend_adx,
    weak_trend_adx=settings.scoring.weak_trend_adx,
)

# 5. Recommendation Builder
recommendation_builder = RecommendationBuilder(
    default_stop_loss_percent=settings.scoring.default_stop_loss_percent,
    default_risk_reward_ratio=settings.scoring.default_risk_reward_ratio,
    tier_strong_threshold=settings.scoring.tier_strong_threshold,
    tier_moderate_threshold=settings.scoring.tier_moderate_threshold,
    tier_weak_threshold=settings.scoring.tier_weak_threshold,
    position_size_strong=settings.scoring.position_size_strong,
    position_size_moderate=settings.scoring.position_size_moderate,
    position_size_weak=settings.scoring.position_size_weak,
)

# Compose into SignalScorer
signal_scorer = SignalScorer(
    credibility_manager=credibility_manager,
    time_calculator=time_calculator,
    confluence_detector=confluence_detector,
    weight_calculator=weight_calculator,
    recommendation_builder=recommendation_builder,
)
```

### AlpacaClient Sharing

Single AlpacaClient instance shared across multiple components:

```python
# Initialize once
alpaca_client = AlpacaClient(
    api_key=settings.alpaca.api_key,
    secret_key=settings.alpaca.secret_key,
    paper=settings.alpaca.paper,
)

# Shared by:
# - MarketGate (for SPY/QQQ volume, VIX checks)
# - TechnicalValidator (for fetching bars, options data)
# - TradeExecutor (for submitting orders)
```

### TradingOrchestrator Assembly

```python
orchestrator = TradingOrchestrator(
    collector_manager=collector_manager,
    analyzer_manager=analyzer_manager,
    technical_validator=technical_validator,
    signal_scorer=signal_scorer,
    risk_manager=risk_manager,
    market_gate=market_gate,
    trade_executor=trade_executor,
    settings=settings.orchestrator,  # OrchestratorSettings from YAML
)
```

---

## Graceful Shutdown & Cleanup

Shutdown happens in reverse order when system stops (Ctrl+C or error):

```python
try:
    await orchestrator.start()
    # System runs here until interrupted

except KeyboardInterrupt:
    print("\nShutting down gracefully...")

finally:
    # Phase 1: Stop orchestrator
    await orchestrator.stop()
    # - Sets state to STOPPING
    # - Cancels background tasks
    # - Processes remaining buffered messages
    # - Disconnects all collectors
    # - Sets state to STOPPED

    # Phase 2: Disconnect infrastructure
    await alpaca_client.disconnect()

    # Phase 3: Final notifications
    if telegram_notifier:
        await telegram_notifier.send_alert("System shutdown complete")

    print("✓ Shutdown complete")
```

### Data Preservation

- Journal entries written to disk immediately (no data loss)
- Tracked positions saved in RiskManager state
- Circuit breaker state persisted

---

## Runtime Behavior & Observability

### Console Output

**Startup:**
```
[2026-01-18 10:00:00] Starting Intraday Trading System in paper mode
[2026-01-18 10:00:01] ✓ Alpaca connected (Cash: $100,000.00)
[2026-01-18 10:00:02] ✓ Collectors ready: Stocktwits, Alpaca News, Reddit
[2026-01-18 10:00:03] ✓ TradingOrchestrator started
```

**Message Flow:**
```
[10:05:23] [Stocktwits] @trader123: NVDA looking bullish
[10:05:24]   → Sentiment: bullish (0.87 confidence)
[10:05:25]   → High-signal detected, processing immediately
[10:05:26]   → Technical validation: PASS
[10:05:27]   → Score: 82 (STRONG) - Entry: $520.50
[10:05:28]   → Risk approved: $500 position
[10:05:29]   → Gate: OPEN
[10:05:30]   ✓ EXECUTED: BUY 0.96 shares NVDA @ $520.50
```

**Batch Processing:**
```
[10:10:00] Processing batch: 23 messages
[10:10:01]   → TSLA: 8 messages, consensus bullish (75%)
[10:10:02]   → AMD: 5 messages, consensus bearish (60%)
[10:10:03]   → AAPL: 10 messages, no consensus (50%)
```

**Circuit Breaker:**
```
[14:30:00] ⚠️  CIRCUIT BREAKER TRIGGERED
[14:30:00]   → Daily loss: -$310 (> $300 limit)
[14:30:00]   → Trading suspended for 60 minutes
```

### Telegram Notifications

- 🚀 New high-confidence signal detected
- ✅ Trade executed: BUY NVDA @ $520.50
- 🛑 Circuit breaker triggered
- 📊 Daily summary (end of day)
- ☑️ Pre-market checklist (9:00 AM)

### Journal Auto-Logging

- Every trade entry → `data/trades/entry_YYYYMMDD_HHMMSS.json`
- Every trade exit → Updates entry file with exit data
- Metrics calculated on-demand via `JournalManager.calculate_metrics()`

### Log Levels

- **INFO**: Normal operations, trade executions
- **WARNING**: Skipped signals, degraded components
- **ERROR**: Connection failures, validation errors

---

## Pre-Launch Configuration

### 1. Environment Variables (`.env`)

**Already Configured:**
```bash
ALPACA_API_KEY=PKA7VO45YLLLP5UG55DZNWOVPA
ALPACA_SECRET_KEY=4EFdKm1sXUSUPwW5SyZ387bSsLE6RmpG3UH4EzAh8ipN
ALPACA_PAPER=true
ANTHROPIC_API_KEY=sk-ant-api03-H0UIn...
TELEGRAM_BOT_TOKEN=8347204323:AAGwd5c4x50dy5ybFx1JWsAAl4DRb4A3fSI
TELEGRAM_CHAT_ID=539691051
```

**Optional (skip if not ready):**
```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=TradingBot/1.0
```

### 2. Create Data Directories

```bash
mkdir -p data/trades data/signals data/cache data/backtest_results
```

### 3. Settings Validation

- `config/settings.yaml` exists ✓
- All sections present ✓

---

## Execution

### Start the System

```bash
uv run python main.py
```

**Expected Output:**
```
Starting Intraday Trading System in paper mode
✓ Settings loaded
✓ Alpaca connected (Cash: $100,000.00)
✓ Telegram connected
✓ FinTwitBERT loaded
✓ Claude API verified
✓ Collectors: Stocktwits, Alpaca News (2/3)
✓ TradingOrchestrator started

Listening for signals...
```

---

## Testing Recommendations

### Phase 1: Dry Run (First 30 minutes)

- Watch console output for any errors
- Verify Telegram receives startup message
- Confirm collectors are streaming messages
- Check that signals are being analyzed

### Phase 2: First Signal (Within 1 hour)

- Wait for first high-confidence signal
- Verify full pipeline execution
- Check journal entry created
- Confirm Telegram notification sent

### Phase 3: Paper Trading (1-2 weeks)

- Let system run during market hours
- Monitor daily for issues
- Review journal metrics weekly
- Adjust thresholds in `settings.yaml` as needed

---

## Success Criteria

✅ System starts without errors
✅ All collectors stream messages
✅ Sentiment analysis processes messages
✅ Technical validation executes
✅ Scoring produces recommendations
✅ Risk checks approve/reject trades
✅ Market gate opens/closes correctly
✅ Trades execute on Alpaca paper account
✅ Journal logs all trades
✅ Telegram sends notifications
✅ Circuit breakers trigger on losses
✅ System shuts down gracefully

---

## Implementation Notes

### Import Organization

```python
# Standard library
import asyncio
import os
from pathlib import Path

# Third-party
from dotenv import load_dotenv

# Project imports (organized by layer)
from src.config.settings import Settings
from src.collectors import CollectorManager, StocktwitsCollector, RedditCollector
from src.analyzers import AnalyzerManager, SentimentAnalyzer, ClaudeAnalyzer
from src.validators import TechnicalValidator
from src.scoring import (
    SignalScorer,
    SourceCredibilityManager,
    TimeFactorCalculator,
    ConfluenceDetector,
    DynamicWeightCalculator,
    RecommendationBuilder,
)
from src.gate import MarketGate
from src.risk import RiskManager
from src.execution import AlpacaClient, TradeExecutor
from src.notifications import TelegramNotifier, AlertFormatter
from src.journal import JournalManager
from src.orchestrator import TradingOrchestrator
```

### Helper Functions

Create helper functions for:
- `validate_env_vars()` - Check required env vars exist
- `create_data_dirs()` - Create data directories
- `print_startup_banner()` - Display system info
- `print_shutdown_summary()` - Display final stats

### Error Recovery

No automatic retry logic - fail fast and let user fix issues. This is clearer for development and prevents cascading failures.

---

## Next Steps

After this design is approved:

1. Create implementation plan (step-by-step tasks)
2. Use git worktree for isolated development
3. Implement main.py integration
4. Test with paper trading account
5. Document any issues or adjustments needed

---

*Design approved: 2026-01-18*
