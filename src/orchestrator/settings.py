"""Configuration for trading orchestrator."""

from pydantic import BaseModel, Field


class OrchestratorSettings(BaseModel):
    """Settings for TradingOrchestrator."""

    enabled: bool = True
    immediate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    batch_interval_seconds: int = Field(default=60, ge=1)
    min_consensus: float = Field(default=0.6, ge=0.0, le=1.0)
    max_buffer_size: int = Field(default=1000, ge=1)
    continue_without_validator: bool = True
    gate_fail_safe_closed: bool = True
