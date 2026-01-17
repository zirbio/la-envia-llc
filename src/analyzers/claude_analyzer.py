# src/analyzers/claude_analyzer.py
import json
import time
from typing import Optional

from anthropic import Anthropic

from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage


class ClaudeAnalyzer:
    """Deep analysis using Claude API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 1000

    ANALYSIS_PROMPT = '''Analyze this social media post about a stock trade opportunity.

Post:
Source: {source}
Author: {author}
Content: {content}
Tickers mentioned: {tickers}

Analyze and respond with a JSON object containing:
- catalyst_type: One of [institutional_flow, earnings, breaking_news, technical_breakout, sector_rotation, insider_activity, fda_approval, merger_acquisition, analyst_upgrade, short_squeeze, unknown]
- catalyst_confidence: Float 0-1 indicating confidence in catalyst identification
- risk_level: One of [low, medium, high, extreme]
- risk_factors: List of specific risk factors identified
- context_summary: Brief summary of the context and why this might be significant
- recommendation: Your recommendation (valid_catalyst, needs_verification, skip, or similar)
- reasoning: Brief explanation of your analysis

Respond ONLY with valid JSON, no other text.'''

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        rate_limit_per_minute: int = 20,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_call_time: Optional[float] = None

    def analyze(self, message: SocialMessage) -> ClaudeAnalysisResult:
        """Analyze a social message for catalyst and risk assessment."""
        self._enforce_rate_limit()

        try:
            tickers = message.extract_tickers()
            prompt = self.ANALYSIS_PROMPT.format(
                source=message.source.value,
                author=message.author,
                content=message.content,
                tickers=", ".join(tickers) if tickers else "None detected",
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            result_data = json.loads(result_text)

            return ClaudeAnalysisResult(
                catalyst_type=CatalystType(result_data["catalyst_type"]),
                catalyst_confidence=result_data["catalyst_confidence"],
                risk_level=RiskLevel(result_data["risk_level"]),
                risk_factors=result_data.get("risk_factors", []),
                context_summary=result_data["context_summary"],
                recommendation=result_data["recommendation"],
                reasoning=result_data.get("reasoning"),
            )

        except Exception:
            return ClaudeAnalysisResult(
                catalyst_type=CatalystType.UNKNOWN,
                catalyst_confidence=0.0,
                risk_level=RiskLevel.HIGH,
                risk_factors=["analysis_failed"],
                context_summary="Analysis failed",
                recommendation="skip",
                reasoning="API error or parsing failure",
            )

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        if self._last_call_time is not None:
            min_interval = 60.0 / self.rate_limit_per_minute
            elapsed = time.time() - self._last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()
