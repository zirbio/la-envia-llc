# tests/config/test_research_settings.py

import pytest
from src.config.settings import Settings, ResearchConfig


class TestResearchConfig:
    def test_research_config_exists(self):
        assert ResearchConfig is not None

    def test_research_config_fields(self):
        config = ResearchConfig(
            enabled=True,
            timezone="Europe/Madrid",
            initial_brief_time="12:00",
            pre_open_brief_time="15:00",
            claude_model="claude-sonnet-4-20250514",
            max_tokens=4000,
            max_ideas=5,
            max_watchlist=5,
            briefs_dir="data/research/briefs",
            inject_to_orchestrator=True,
            telegram_enabled=True,
            telegram_summary=True,
        )

        assert config.enabled is True
        assert config.timezone == "Europe/Madrid"
        assert config.initial_brief_time == "12:00"

    def test_settings_has_research(self):
        # This will fail until we add research to Settings
        settings = Settings.from_yaml("config/settings.yaml")
        assert hasattr(settings, "research")
