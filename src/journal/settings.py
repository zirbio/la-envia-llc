# src/journal/settings.py
"""Settings for the journal module."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


WeekDay = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class JournalSettings(BaseModel):
    """Configuration settings for the trading journal.

    Attributes:
        enabled: Whether journaling is enabled.
        data_dir: Directory to store trade JSON files.
        auto_log_entries: Auto-log when trades open.
        auto_log_exits: Auto-log when trades close.
        default_period_days: Default period for metrics calculation.
        weekly_report_enabled: Generate weekly reports.
        weekly_report_day: Day to generate weekly report.
    """

    enabled: bool = True
    data_dir: str = "data/trades"

    auto_log_entries: bool = True
    auto_log_exits: bool = True

    default_period_days: int = Field(default=30, ge=1, le=365)

    weekly_report_enabled: bool = True
    weekly_report_day: WeekDay = "saturday"

    @field_validator("weekly_report_day")
    @classmethod
    def validate_weekday(cls, v: str) -> str:
        """Validate that weekly_report_day is a valid weekday."""
        valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        if v.lower() not in valid_days:
            raise ValueError(f"Invalid weekday: {v}. Must be one of {valid_days}")
        return v.lower()
