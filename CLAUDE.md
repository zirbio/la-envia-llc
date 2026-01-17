# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered intraday trading system that combines social media analysis (Twitter, Reddit, Stocktwits), sentiment analysis (FinTwitBERT + Claude), technical validation, and automated execution via Alpaca API.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests (664 tests)
uv run pytest

# Run single test file
uv run pytest tests/path/test_file.py -v

# Run specific test
uv run pytest tests/path/test_file.py::test_function_name -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Start main system
uv run python main.py

# Start dashboard
uv run streamlit run src/dashboard/Home.py --server.port 8501

# Lint (ruff)
uv run ruff check src/

# Format (black)
uv run black src/
```

## Architecture

7-layer pipeline processing social signals into trade execution:

```
Layer 0: Market Gate (src/gate/)
    → Trading hours, SPY/QQQ volume, VIX levels, choppy detection

Layer 1: Collectors (src/collectors/)
    → Twitter (twscrape), Reddit (asyncpraw), Stocktwits, Alpaca News

Layer 2: Analyzers (src/analyzers/)
    → FinTwitBERT sentiment, Claude AI catalyst/risk analysis

Layer 3: Technical Validation (src/validators/)
    → RSI, MACD, ADX, volume, options flow, IV rank

Layer 4: Scoring (src/scoring/)
    → SignalScorer: sentiment(50%) + technical(50%) + bonuses

Layer 5: Risk Management (src/risk/)
    → Circuit breakers (per-trade 1%, daily 3%, weekly 6%)

Layer 6: Execution (src/execution/)
    → TradeExecutor via Alpaca API + Journal logging
```

**Orchestrator** (`src/orchestrator/`) coordinates the pipeline with hybrid processing: immediate execution for high-confidence signals (≥85%), batch processing for others.

## Key Modules

| Module | Entry Point | Purpose |
|--------|-------------|---------|
| `scoring` | `SignalScorer` | Calculate final score, build `TradeRecommendation` |
| `execution` | `TradeExecutor` | Submit orders, track positions |
| `validators` | `TechnicalValidator` | Validate signals against technical indicators |
| `gate` | `MarketGate` | Check market conditions before trading |
| `risk` | `RiskManager` | Evaluate trades, enforce circuit breakers |
| `orchestrator` | `TradingOrchestrator` | Coordinate full pipeline |
| `journal` | `JournalManager` | Log trades, calculate metrics |

## Configuration

All tunable parameters in `config/settings.yaml`. Key sections:
- `scoring`: thresholds (strong≥80, moderate≥60), weights, bonuses
- `risk`: circuit breaker limits (per-trade, daily, weekly)
- `validators.technical`: indicator periods (RSI=14, MACD=12/26/9)
- `market_gate`: trading hours, volume minimums, VIX limits

API credentials via `.env`:
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER`
- `ANTHROPIC_API_KEY`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## Code Patterns

- **Async-first**: All I/O uses `asyncio` patterns
- **Pydantic models**: Data validation via `src/models/` (e.g., `SocialMessage`, `TradeRecommendation`)
- **Settings loader**: `Settings.from_yaml()` in `src/config/settings.py`
- **Tests mirror source**: `tests/` structure matches `src/`
- **Validation scenarios**: `tests/validation/` and `src/validation/` for end-to-end testing

## Testing Notes

- 664 tests across unit, integration, and validation categories
- Use realistic mock data from validation scenarios, not random fixtures
- Integration tests in `tests/integration/` test complete data flows
- All async tests use `pytest-asyncio` with `asyncio_mode = "auto"`

## Documentation

User documentation in Spanish:
- `docs/MANUAL-DE-USO.md` - Complete usage manual
- `docs/SIGUIENTES-PASOS.md` - Setup and configuration steps
- `docs/plans/` - Design documents for each implementation phase
