from pydantic import BaseModel, Field


class ValidationSettings(BaseModel):
    """Configuration for validation and simulation."""

    scenario_timeout_seconds: int = Field(default=30, gt=0)
    mock_market_delay_ms: int = Field(default=100, ge=0)
    fail_fast: bool = True
