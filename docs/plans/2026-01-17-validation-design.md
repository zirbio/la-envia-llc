# Phase 12: System Validation - Design Document

**Goal:** Comprehensive system validation to verify all components work together correctly.

**Scope:** Integration tests per module boundary + full E2E simulation with complete coverage (happy paths + error/edge scenarios).

---

## Architecture

The validation module provides two testing layers:

### 1. Integration Tests (`tests/integration/`)
- Test module boundaries with real component instances (mocked external APIs)
- Data flow chain: Collector → Analyzer → Scoring → Validator
- Execution chain: Signal → Gate → Risk → Execution → Journal

### 2. E2E Simulator (`src/validation/`)
- `SimulatorEngine` - Orchestrates full pipeline with mock market data
- `ScenarioRunner` - Executes predefined test scenarios
- `MockMarketData` - Generates realistic price/volume data
- `ValidationReport` - Aggregates results and assertions

### Directory Structure
```
src/validation/
├── __init__.py
├── settings.py
├── simulator_engine.py
├── scenario_runner.py
├── mock_market_data.py
├── scenarios/
│   ├── __init__.py
│   ├── happy_path.py
│   └── error_scenarios.py
└── validation_report.py

tests/integration/
├── __init__.py
├── test_data_flow_chain.py
└── test_execution_chain.py
```

---

## Integration Tests

### Data Flow Chain Tests (`test_data_flow_chain.py`)

Tests the signal generation pipeline:
- `test_collector_to_analyzer` - Social message flows to sentiment analysis
- `test_analyzer_to_scoring` - Analyzed message gets scored correctly
- `test_scoring_to_validator` - Scored signal passes through technical validation
- `test_full_data_flow` - Complete chain from raw message to validated signal

### Execution Chain Tests (`test_execution_chain.py`)

Tests the trade execution pipeline:
- `test_signal_to_gate` - Validated signal checked by market gate
- `test_gate_to_risk` - Gate-passed signal evaluated by risk manager
- `test_risk_to_execution` - Risk-approved signal sent to Alpaca (mocked)
- `test_execution_to_journal` - Executed trade logged correctly
- `test_full_execution_flow` - Complete chain from signal to journal entry

**Key Principle:** Each test uses real component instances with only external APIs mocked (Alpaca, Claude, Twitter). This catches integration bugs that unit tests miss.

---

## E2E Simulator Components

### SimulatorEngine (`simulator_engine.py`)
- Initializes all system components with mock dependencies
- Provides `run_scenario(scenario)` method
- Collects metrics: signals generated, trades executed, errors caught
- Returns `SimulationResult` with pass/fail status

### MockMarketData (`mock_market_data.py`)
- Generates OHLCV data for testing indicators
- Configurable: trending up, trending down, sideways, volatile
- Provides mock Alpaca responses for `get_bars()` calls

### ScenarioRunner (`scenario_runner.py`)
- Loads scenario definitions
- Injects mock social messages into the pipeline
- Asserts expected outcomes (trade executed, gate blocked, etc.)
- Supports timeouts and async operations

### ValidationReport (`validation_report.py`)
- Aggregates results from multiple scenarios
- Tracks: scenarios run, passed, failed, errors
- Generates summary with failure details

---

## Test Scenarios

### Happy Path Scenarios (`happy_path.py`)
- `bullish_signal_executes_trade` - High-confidence bullish signal → buy order executed → logged to journal
- `bearish_signal_executes_trade` - High-confidence bearish signal → sell/short order executed
- `multiple_signals_same_symbol` - Handles concurrent signals correctly

### Error Scenarios (`error_scenarios.py`)
- `gate_blocks_outside_hours` - Market gate rejects signal when market closed
- `risk_limit_blocks_trade` - Daily loss limit reached → trade rejected
- `circuit_breaker_triggers` - Max losses hit → system halts trading
- `low_confidence_filtered` - Weak sentiment signal never reaches execution
- `technical_veto_blocks` - RSI overbought vetoes bullish signal
- `api_failure_handled` - Alpaca API error caught, logged, no crash
- `analyzer_timeout` - Claude timeout handled gracefully

### Scenario Definition
Each scenario defines:
- **Input:** Mock social message + mock market conditions
- **Expected:** Specific outcome (trade, rejection, error handling)
- **Assertions:** Verify journal entries, alerts, system state

---

## Configuration & Running

### Simulator Settings (`src/validation/settings.py`)
```python
class ValidationSettings(BaseModel):
    scenario_timeout_seconds: int = Field(default=30, gt=0)
    mock_market_delay_ms: int = Field(default=100, ge=0)
    fail_fast: bool = True  # Stop on first failure
```

### Running Validation
- Integration tests: `pytest tests/integration/ -v`
- E2E scenarios: `python -m src.validation.scenario_runner`
- Both combined: `pytest tests/ -v --integration --e2e`

### CI/CD Integration
- All validation tests run before merge to main
- Scenario runner produces JSON report for CI parsing
- Exit code 0 = all pass, 1 = failures

**No external dependencies required** - All mocks are self-contained. Tests can run offline without Alpaca/Claude/Twitter credentials.
