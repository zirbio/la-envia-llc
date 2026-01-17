"""Settings for the Streamlit dashboard."""
from typing import Literal

from pydantic import BaseModel, Field


class DashboardSettings(BaseModel):
    """Configuration for the dashboard."""

    refresh_interval_seconds: int = Field(default=5, gt=0)
    max_signals_displayed: int = Field(default=10, gt=0)
    max_alerts_displayed: int = Field(default=50, gt=0)
    theme: Literal["light", "dark"] = "dark"
