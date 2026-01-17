# src/config/settings.py
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    batch_size: int = 32
    min_confidence: float = 0.7


class ClaudeAnalyzerSettings(BaseModel):
    """Settings for Claude analyzer."""

    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1000
    rate_limit_per_minute: int = 20
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
