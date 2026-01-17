"""Tests for OrchestratorSettings configuration."""

import pytest

from config.settings import Settings
from orchestrator.settings import OrchestratorSettings


class TestOrchestratorSettings:
    """Tests for orchestrator settings."""

    def test_default_values(self) -> None:
        settings = OrchestratorSettings()

        assert settings.enabled is True
        assert settings.immediate_threshold == 0.85
        assert settings.batch_interval_seconds == 60
        assert settings.min_consensus == 0.6
        assert settings.max_buffer_size == 1000
        assert settings.continue_without_validator is True
        assert settings.gate_fail_safe_closed is True

    def test_custom_values(self) -> None:
        settings = OrchestratorSettings(
            immediate_threshold=0.9,
            batch_interval_seconds=30,
        )
        assert settings.immediate_threshold == 0.9
        assert settings.batch_interval_seconds == 30

    def test_settings_has_orchestrator_config(self) -> None:
        settings = Settings()
        assert hasattr(settings, "orchestrator")
        assert isinstance(settings.orchestrator, OrchestratorSettings)
