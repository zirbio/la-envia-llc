# src/research/prompts/templates.py

"""Prompt templates for Morning Research Agent."""

SYSTEM_PROMPT = """You are a senior equity research analyst at a quantitative hedge fund.
Your job: produce the Morning Brief before US market open.

PRINCIPLES:
1. Every conclusion must cite evidence (data source + specific numbers)
2. Distinguish FACT (data) from INFERENCE (your analysis)
3. Think contrarian: what is the market missing or mispricing?
4. Express conviction in probabilities, not certainties
5. Focus on asymmetric risk/reward (2:1 minimum)
6. If data is insufficient, say "INSUFFICIENT DATA" - never guess

REJECTION CRITERIA (automatic NO):
- No clear catalyst in next 24-48 hours
- Risk/reward below 1.5:1
- Conflicting signals without resolution
- Low liquidity (< 500K avg volume)"""


def build_context(data: dict) -> str:
    """Build context string from fetched data.

    Args:
        data: Dictionary with fetched market data.

    Returns:
        Formatted context string for the prompt.
    """
    futures = data.get("futures", {})
    gappers = data.get("gappers", [])
    earnings = data.get("earnings", [])
    economic = data.get("economic_events", [])
    sec = data.get("sec_filings", {"8k": [], "form4": []})
    social = data.get("social_intelligence", "No social data available")
    news = data.get("news", [])

    # Format gappers table
    gappers_table = ""
    for g in gappers:
        gappers_table += f"| {g.get('ticker', 'N/A')} | {g.get('gap_percent', 0):.1f}% | {g.get('volume', 0):,} | {g.get('catalyst', 'Unknown')} |\n"
    if not gappers_table:
        gappers_table = "| No significant gappers found |\n"

    return f"""<context>
<market_snapshot>
Date: {data.get('date', 'Unknown')} | Time: {data.get('time', 'Unknown')} ET
ES Futures: {futures.get('es', 'N/A')} ({futures.get('es_change', 0):.1f}%)
NQ Futures: {futures.get('nq', 'N/A')} ({futures.get('nq_change', 0):.1f}%)
VIX: {data.get('vix', 'N/A')}
</market_snapshot>

<premarket_gappers>
| Ticker | Gap% | Volume | Catalyst |
|--------|------|--------|----------|
{gappers_table}</premarket_gappers>

<earnings_today>
{', '.join(earnings) if earnings else 'No major earnings today'}
</earnings_today>

<economic_calendar>
{chr(10).join(economic) if economic else 'No major events scheduled'}
</economic_calendar>

<sec_filings_24h>
8-K Material Events: {len(sec.get('8k', []))} filings
Form 4 Insider Activity: {len(sec.get('form4', []))} filings
</sec_filings_24h>

<social_intelligence>
{social}
</social_intelligence>

<overnight_news>
{chr(10).join(news) if news else 'No significant overnight news'}
</overnight_news>
</context>"""


TASK_PROMPT = """Analyze ALL data above. Produce today's MORNING BRIEF.

REQUIRED SECTIONS:

1. MARKET REGIME (3-4 sentences)
   - Current market state (risk-on/off, trending/ranging)
   - Key overnight developments
   - What the VIX and futures are signaling

2. TOP IDEAS (max 5, ranked by conviction)
   For EACH idea provide:
   - Ticker, Direction (LONG/SHORT), Conviction (HIGH/MED/LOW)
   - Catalyst: What's driving this TODAY? (specific event/data)
   - Thesis: Why will price move in your direction?
   - Technical: Key levels (support, resistance, entry zone)
   - Risk/Reward: Entry, Stop, Target (with R:R ratio)
   - Position Size: FULL (1x) / HALF (0.5x) / QUARTER (0.25x)
   - Kill Switch: What would invalidate this idea?

3. WATCHLIST (3-5 tickers)
   - Setting up but not actionable today
   - What trigger would make them tradeable?

4. RISKS & LANDMINES
   - Events that could blow up positions
   - Scheduled times to be careful

5. KEY QUESTIONS
   - What information would change your thesis today?

Return ONLY valid JSON matching this schema:
{
  "market_regime": {
    "state": "risk-on|risk-off|neutral",
    "trend": "bullish|bearish|ranging",
    "summary": "string"
  },
  "ideas": [
    {
      "rank": 1,
      "ticker": "NVDA",
      "direction": "LONG",
      "conviction": "HIGH",
      "catalyst": "string",
      "thesis": "string",
      "technical": {
        "support": 135.50,
        "resistance": 142.00,
        "entry_zone": [136.00, 137.50]
      },
      "risk_reward": {
        "entry": 137.00,
        "stop": 134.50,
        "target": 145.00,
        "ratio": "3.2:1"
      },
      "position_size": "FULL",
      "kill_switch": "string"
    }
  ],
  "watchlist": [
    {"ticker": "MSFT", "setup": "string", "trigger": "string"}
  ],
  "risks": ["string"],
  "key_questions": ["string"]
}"""
