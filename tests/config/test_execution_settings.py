# tests/config/test_execution_settings.py
import pytest
from src.config.settings import ExecutionSettings, Settings


class TestExecutionSettings:
    def test_default_values(self):
        settings = ExecutionSettings()
        assert settings.enabled is True
        assert settings.paper_mode is True
        assert settings.default_time_in_force == "day"

    def test_custom_values(self):
        settings = ExecutionSettings(
            enabled=False,
            paper_mode=False,
            default_time_in_force="gtc",
        )
        assert settings.enabled is False
        assert settings.paper_mode is False
        assert settings.default_time_in_force == "gtc"

    def test_time_in_force_accepts_valid_values(self):
        for tif in ["day", "gtc", "ioc", "fok"]:
            settings = ExecutionSettings(default_time_in_force=tif)
            assert settings.default_time_in_force == tif

    def test_settings_has_execution_config(self):
        settings = Settings()
        assert hasattr(settings, "execution")
        assert isinstance(settings.execution, ExecutionSettings)
