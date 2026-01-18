# tests/research/prompts/test_templates.py

import pytest
from src.research.prompts.templates import (
    SYSTEM_PROMPT,
    build_context,
    TASK_PROMPT,
)


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_principles(self):
        assert "PRINCIPIOS" in SYSTEM_PROMPT
        assert "evidencia" in SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_rejection_criteria(self):
        assert "RECHAZO" in SYSTEM_PROMPT


class TestBuildContext:
    def test_build_context_with_data(self):
        data = {
            "date": "2026-01-18",
            "time": "06:00",
            "futures": {"es": 5000.0, "nq": 17500.0, "es_change": 0.5, "nq_change": 0.8},
            "vix": 14.5,
            "gappers": [{"ticker": "NVDA", "gap_percent": 5.0, "volume": 500000}],
            "earnings": ["NFLX", "TSLA"],
            "economic_events": ["CPI at 14:30"],
            "sec_filings": {"8k": [], "form4": []},
            "social_intelligence": "Unusual activity in NVDA options",
            "news": ["Tech earnings strong"],
        }

        context = build_context(data)

        assert "2026-01-18" in context
        assert "5000.0" in context
        assert "NVDA" in context
        assert "CPI" in context

    def test_build_context_handles_missing_data(self):
        data = {"date": "2026-01-18", "time": "06:00"}

        context = build_context(data)

        assert "2026-01-18" in context
        # Should not raise, should use defaults


class TestTaskPrompt:
    def test_task_prompt_exists(self):
        assert TASK_PROMPT is not None
        assert len(TASK_PROMPT) > 100

    def test_task_prompt_contains_sections(self):
        assert "RÉGIMEN DE MERCADO" in TASK_PROMPT
        assert "IDEAS PRINCIPALES" in TASK_PROMPT
        assert "WATCHLIST" in TASK_PROMPT
        assert "RIESGOS" in TASK_PROMPT

    def test_task_prompt_contains_json_schema(self):
        assert "output_format" in TASK_PROMPT.lower() or "json" in TASK_PROMPT.lower()
