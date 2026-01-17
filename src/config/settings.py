# src/config/settings.py
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gate.market_gate import MarketGateSettings
from orchestrator.settings import OrchestratorSettings


class SystemConfig(BaseModel):
    name: str = "Intraday Trading System"
    version: str = "1.0.0"
    mode: str = "paper"
    timezone: str = "America/New_York"


class TwitterCollectorConfig(BaseModel):
    enabled: bool = True
    engine: str = "twscrape"
    accounts_pool_size: int = 5
    rate_limit_buffer: float = 0.8
    refresh_interval_seconds: int = 15


class RedditCollectorConfig(BaseModel):
    enabled: bool = True
    use_streaming: bool = True
    batch_fallback_interval: int = 60


class StocktwitsCollectorConfig(BaseModel):
    enabled: bool = True
    refresh_interval_seconds: int = 30


class CollectorsConfig(BaseModel):
    twitter: TwitterCollectorConfig = Field(default_factory=TwitterCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    stocktwits: StocktwitsCollectorConfig = Field(default_factory=StocktwitsCollectorConfig)


class PerTradeRiskConfig(BaseModel):
    max_loss_percent: float = 1.0
    hard_stop: bool = True


class DailyRiskConfig(BaseModel):
    max_loss_percent: float = 3.0
    max_trades_after_loss: int = 0
    cooldown_minutes: int = 60


class WeeklyRiskConfig(BaseModel):
    max_loss_percent: float = 6.0
    force_paper_mode: bool = True


class CircuitBreakersConfig(BaseModel):
    per_trade: PerTradeRiskConfig = Field(default_factory=PerTradeRiskConfig)
    daily: DailyRiskConfig = Field(default_factory=DailyRiskConfig)
    weekly: WeeklyRiskConfig = Field(default_factory=WeeklyRiskConfig)


class RiskConfig(BaseModel):
    circuit_breakers: CircuitBreakersConfig = Field(default_factory=CircuitBreakersConfig)


class SentimentAnalyzerSettings(BaseModel):
    """Settings for sentiment analyzer."""

    model: str = "StephanAkkerman/FinTwitBERT-sentiment"
    batch_size: int = Field(default=32, ge=1, le=256)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ClaudeAnalyzerSettings(BaseModel):
    """Settings for Claude analyzer."""

    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = Field(default=1000, ge=100, le=4096)
    rate_limit_per_minute: int = Field(default=20, gt=0, le=100)
    use_for: list[str] = Field(
        default_factory=lambda: [
            "catalyst_classification",
            "risk_assessment",
            "context_analysis",
        ]
    )


class AnalyzersSettings(BaseModel):
    """Settings for all analyzers."""

    sentiment: SentimentAnalyzerSettings = Field(default_factory=SentimentAnalyzerSettings)
    claude: ClaudeAnalyzerSettings = Field(default_factory=ClaudeAnalyzerSettings)


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

    technical: TechnicalValidatorSettings = Field(default_factory=TechnicalValidatorSettings)


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
    tier1_sources: list[str] = Field(
        default_factory=lambda: [
            "unusual_whales",
            "optionsflow",
        ]
    )


class RiskSettings(BaseModel):
    """Settings for risk management."""

    enabled: bool = True
    max_position_value: float = Field(default=1000.0, gt=0)
    max_daily_loss: float = Field(default=500.0, gt=0)
    unrealized_warning_threshold: float = Field(default=300.0, gt=0)


class ExecutionSettings(BaseModel):
    """Settings for trade execution."""

    enabled: bool = True
    paper_mode: bool = True
    default_time_in_force: str = Field(default="day")


class AlpacaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPACA_")

    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    paper_url: str = "https://paper-api.alpaca.markets"
    live_url: str = "https://api.alpaca.markets"


class RedditAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDDIT_")

    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "TradingBot/1.0"


class Settings(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    analyzers: AnalyzersSettings = Field(default_factory=AnalyzersSettings)
    validators: ValidatorsSettings = Field(default_factory=ValidatorsSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    risk_settings: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    market_gate: MarketGateSettings = Field(default_factory=MarketGateSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    reddit_api: RedditAPIConfig = Field(default_factory=RedditAPIConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file with env var overrides."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        alpaca = AlpacaConfig()
        reddit_api = RedditAPIConfig()

        return cls(
            **data,
            alpaca=alpaca,
            reddit_api=reddit_api,
        )
