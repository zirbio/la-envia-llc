# Morning Research Agent - Diseño

**Fecha:** 2026-01-18
**Estado:** Aprobado
**Propósito:** Añadir análisis profundo pre-market que complementa el sistema de Grok

---

## Resumen Ejecutivo

El Morning Research Agent es un módulo independiente que genera un Daily Brief antes de la apertura del mercado estadounidense. Combina múltiples fuentes de datos (Grok, SEC, Yahoo, Alpaca) y usa Claude para sintetizar análisis profundos que se inyectan al sistema de trading como señales.

### Decisiones de Diseño

| Aspecto | Decisión |
|---------|----------|
| Rol | Complementar sistema Grok (no reemplazar) |
| Análisis | Fundamentales + Macro + Flow (completo) |
| LLM | Híbrido: Grok fetch → Claude analyze |
| Horario | 12:00 + 15:00 hora Almería (6:00 + 9:00 AM ET) |
| Output | Telegram + Dashboard + JSON |
| Integración | Señales sintéticas con scoring integrado |
| Arquitectura | Módulo independiente (Enfoque A) |

### Beneficios

| Aspecto | Sistema Actual | Con Research Agent |
|---------|----------------|-------------------|
| Enfoque | Reactivo (espera tweets) | Proactivo + Reactivo |
| Profundidad | Tweet individual | Análisis multi-fuente |
| Cobertura | Solo X/Twitter | SEC + Earnings + Macro + Flow |
| Decisiones | Basadas en sentiment | Basadas en thesis completa |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MORNING RESEARCH AGENT                            │
│                    (12:00 + 15:00 hora Almería)                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  GROK FETCH   │    │ FREE APIs     │    │  CLAUDE       │
│               │    │               │    │  SYNTHESIS    │
│ • X/Twitter   │    │ • SEC EDGAR   │    │               │
│ • Web news    │    │ • Yahoo Fin   │    │ Prompt mega   │
│ • Sentiment   │    │ • Finviz gaps │    │ curado que    │
│               │    │ • Econ cal    │    │ genera brief  │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   DailyBrief    │
                    │                 │
                    │ • market_context│
                    │ • ideas[]       │
                    │ • watchlist[]   │
                    │ • risks[]       │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Telegram   │    │    Dashboard    │    │  Trading    │
│  (resumen)  │    │  (página nueva) │    │   System    │
│             │    │                 │    │             │
│             │    │                 │    │ ideas[] →   │
│             │    │                 │    │ SocialMsg   │
│             │    │                 │    │ (RESEARCH)  │
└─────────────┘    └─────────────────┘    └─────────────┘
```

---

## Estructura de Archivos

```
src/research/                    # NUEVO módulo
├── __init__.py
├── morning_agent.py             # Orquestador principal
├── integration.py               # Conversión a SocialMessage
├── data_fetchers/
│   ├── __init__.py
│   ├── base.py                  # BaseFetcher abstract
│   ├── grok_fetcher.py          # X/Twitter + Web via Grok
│   ├── sec_fetcher.py           # SEC EDGAR 8-K, Form 4
│   ├── market_fetcher.py        # Futures, VIX, gappers
│   ├── earnings_fetcher.py      # Earnings calendar
│   ├── economic_fetcher.py      # Economic calendar
│   └── news_fetcher.py          # Alpaca News
├── prompts/
│   ├── __init__.py
│   ├── system_prompt.py         # Persona + principios
│   ├── context_template.py      # Template de contexto
│   └── task_template.py         # Task + output format
└── models/
    ├── __init__.py
    ├── trading_idea.py          # TradingIdea model
    ├── daily_brief.py           # DailyBrief model
    └── market_context.py        # MarketContext (datos crudos)

src/dashboard/pages/
└── 5_Research.py                # Nueva página dashboard

data/research/                   # NUEVO directorio
└── briefs/
    ├── 2026-01-18_initial.json
    ├── 2026-01-18_pre_open.json
    └── ...
```

---

## Data Fetchers

| Fetcher | Datos | Fuente | Costo |
|---------|-------|--------|-------|
| **Grok** | Tweets de traders, noticias trending, sentiment | xAI API | Ya disponible |
| **SEC** | 8-K (eventos materiales), Form 4 (insider trading) | SEC EDGAR | Gratis |
| **Market** | ES/NQ futures, VIX, gappers >3% | Yahoo Finance | Gratis |
| **Earnings** | Empresas que reportan hoy/mañana | Yahoo Finance | Gratis |
| **Economic** | FOMC, CPI, Jobs, GDP releases | Finnhub/Investing | Gratis |
| **News** | Breaking news overnight | Alpaca News API | Ya disponible |

### Grok Fetcher Queries

```python
queries = [
    "most discussed stocks pre-market",
    "unusual options activity today",
    "earnings surprises",
    "from:unusual_whales",
    "from:DeItaone",
    "from:FirstSquawk",
]
```

---

## Prompt de Claude

### System Prompt

```xml
<system>
You are a senior equity research analyst at a quantitative hedge fund.
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
- Low liquidity (< 500K avg volume)
</system>
```

### Context Template

```xml
<context>
<market_snapshot>
Date: {date} | Time: {time} ET
ES Futures: {es} ({es_change}%) | NQ: {nq} ({nq_change}%)
VIX: {vix} | 10Y Yield: {yield_10y}
Asia: {asia_summary} | Europe: {europe_summary}
</market_snapshot>

<premarket_gappers>
| Ticker | Gap% | Volume | Catalyst |
|--------|------|--------|----------|
{gappers_table}
</premarket_gappers>

<earnings_today>
Before Open: {earnings_bmo}
After Close: {earnings_amc}
</earnings_today>

<economic_calendar>
{economic_events}
</economic_calendar>

<sec_filings_24h>
8-K Material Events:
{eight_k_filings}

Form 4 Insider Activity:
{form_4_filings}
</sec_filings_24h>

<social_intelligence>
{grok_insights}
</social_intelligence>

<overnight_news>
{alpaca_news}
</overnight_news>
</context>
```

### Task Template

```xml
<task>
Analyze ALL data above. Produce today's MORNING BRIEF.

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
</task>

<output_format>
Return ONLY valid JSON matching this schema:
{
  "generated_at": "ISO timestamp",
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
      "catalyst": "string (specific, dated)",
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
      "kill_switch": "string (what invalidates this)"
    }
  ],
  "watchlist": [...],
  "risks": [...],
  "key_questions": [...]
}
</output_format>
```

---

## Modelos de Datos

### TradingIdea

```python
# src/research/models/trading_idea.py

from pydantic import BaseModel
from enum import Enum

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class Conviction(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class PositionSize(str, Enum):
    FULL = "FULL"        # 1.0x normal size
    HALF = "HALF"        # 0.5x
    QUARTER = "QUARTER"  # 0.25x

class TechnicalLevels(BaseModel):
    support: float
    resistance: float
    entry_zone: tuple[float, float]

class RiskReward(BaseModel):
    entry: float
    stop: float
    target: float
    ratio: str  # "2.5:1"

class TradingIdea(BaseModel):
    rank: int
    ticker: str
    direction: Direction
    conviction: Conviction
    catalyst: str
    thesis: str
    technical: TechnicalLevels
    risk_reward: RiskReward
    position_size: PositionSize
    kill_switch: str
```

### DailyBrief

```python
# src/research/models/daily_brief.py

class MarketRegime(BaseModel):
    state: Literal["risk-on", "risk-off", "neutral"]
    trend: Literal["bullish", "bearish", "ranging"]
    summary: str

class WatchlistItem(BaseModel):
    ticker: str
    setup: str
    trigger: str

class DailyBrief(BaseModel):
    generated_at: datetime
    brief_type: Literal["initial", "pre_open"]
    timezone: str = "Europe/Madrid"

    market_regime: MarketRegime
    ideas: list[TradingIdea]
    watchlist: list[WatchlistItem]
    risks: list[str]
    key_questions: list[str]

    # Metadata
    data_sources_used: list[str]
    fetch_duration_seconds: float
    analysis_duration_seconds: float
```

---

## Integración con Sistema Existente

### Nuevo SourceType

```python
# src/models/social_message.py (modificar)

class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    GROK = "grok"
    RESEARCH = "research"  # ← NUEVO
```

### Conversión TradingIdea → SocialMessage

```python
# src/research/integration.py

def idea_to_social_message(idea: TradingIdea, brief: DailyBrief) -> SocialMessage:
    """Convierte una idea del brief en SocialMessage para el sistema."""

    content = (
        f"${idea.ticker} {idea.direction.value} - {idea.conviction.value} conviction. "
        f"Catalyst: {idea.catalyst}. "
        f"Entry: ${idea.risk_reward.entry}, Stop: ${idea.risk_reward.stop}, "
        f"Target: ${idea.risk_reward.target} (R:R {idea.risk_reward.ratio}). "
        f"Kill switch: {idea.kill_switch}"
    )

    return SocialMessage(
        source=SourceType.RESEARCH,
        source_id=f"brief_{brief.generated_at.isoformat()}_{idea.ticker}",
        author="morning_research_agent",
        content=content,
        timestamp=brief.generated_at,
        url=None,
        metadata={
            "rank": idea.rank,
            "conviction": idea.conviction.value,
            "direction": idea.direction.value,
            "entry": idea.risk_reward.entry,
            "stop": idea.risk_reward.stop,
            "target": idea.risk_reward.target,
            "position_size": idea.position_size.value,
        }
    )
```

### Inyección al Orchestrator

```python
# src/research/morning_agent.py

class MorningResearchAgent:
    async def inject_to_trading_system(
        self,
        brief: DailyBrief,
        orchestrator: TradingOrchestrator
    ) -> None:
        """Inyecta las ideas del brief al sistema de trading."""

        for idea in brief.ideas:
            msg = idea_to_social_message(idea, brief)
            await orchestrator.process_message(msg)

            logger.info(
                f"Injected {idea.ticker} ({idea.direction}) "
                f"conviction={idea.conviction} to trading system"
            )
```

### Tracking de Accuracy

El `DynamicCredibilityManager` trackeará `morning_research_agent` como cualquier otra fuente:

```
Señal: NVDA LONG @ $137
     ↓
30 min después: NVDA @ $142 (+3.6%)
     ↓
SignalOutcomeTracker: was_correct = True
     ↓
DynamicCredibilityManager.record_outcome("morning_research_agent", True)
     ↓
Con el tiempo: accuracy 70% → multiplier 1.3x
```

---

## Dashboard

Nueva página: `src/dashboard/pages/5_Research.py`

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 MORNING BRIEF - 18 Ene 2026 (15:00 Pre-Open)               │
│  Market: RISK-ON | Trend: BULLISH | VIX: 14.2                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🎯 TOP IDEAS                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ #1 NVDA LONG ⬆️ [HIGH] ████████████ Full Position      │   │
│  │ Catalyst: TSMC beat + AI guidance raise                 │   │
│  │ Entry: $136-137.50 | Stop: $134.50 | Target: $145      │   │
│  │ R:R 3.2:1 | Kill: China export headline                │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ #2 AMD LONG ⬆️ [MEDIUM] ████████ Half Position         │   │
│  │ ...                                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  👀 WATCHLIST                     ⚠️ RISKS                     │
│  ┌─────────────────────┐         ┌─────────────────────────┐   │
│  │ MSFT - Earnings Wed │         │ • 14:30 CPI release     │   │
│  │ META - Setting up   │         │ • FOMC minutes 20:00    │   │
│  └─────────────────────┘         └─────────────────────────┘   │
│                                                                 │
│  ❓ KEY QUESTIONS                                               │
│  • Will NVDA hold $135 support on any pullback?               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  📈 RESEARCH AGENT PERFORMANCE                                  │
│  Last 30 days: 67% accuracy | 45 signals | Multiplier: 1.1x   │
└─────────────────────────────────────────────────────────────────┘
```

### Funcionalidades

- Selector de fecha para briefs históricos
- Toggle Initial/Pre-Open
- Colores por convicción (HIGH=verde, MEDIUM=amarillo, LOW=gris)
- Métricas del Agent (accuracy histórica)
- Auto-refresh cuando hay nuevo brief

---

## Configuración

Añadir a `config/settings.yaml`:

```yaml
# Morning Research Agent
research:
  enabled: true

  # Horarios (hora local Almería)
  timezone: "Europe/Madrid"
  schedule:
    initial_brief: "12:00"    # 6:00 AM ET
    pre_open_brief: "15:00"   # 9:00 AM ET

  # LLM providers
  llm:
    fetcher: "grok"
    analyzer: "claude"
    claude_model: "claude-sonnet-4-20250514"
    max_tokens: 4000

  # Data fetchers
  fetchers:
    grok:
      enabled: true
      queries:
        - "most discussed stocks pre-market"
        - "unusual options activity today"
        - "earnings surprises"
        - "from:unusual_whales"
        - "from:DeItaone"
        - "from:FirstSquawk"

    sec_edgar:
      enabled: true
      forms: ["8-K", "4"]
      lookback_hours: 24

    market:
      enabled: true
      min_gap_percent: 3.0
      min_volume: 100000

    earnings:
      enabled: true
      days_ahead: 2

    economic:
      enabled: true
      min_impact: "medium"

    alpaca_news:
      enabled: true
      lookback_hours: 12

  # Output
  output:
    max_ideas: 5
    max_watchlist: 5
    save_briefs: true
    briefs_dir: "data/research/briefs"

  # Integración con trading system
  integration:
    inject_to_orchestrator: true
    author_id: "morning_research_agent"

  # Notificaciones
  notifications:
    telegram_enabled: true
    telegram_summary: true
    telegram_full_brief: false
```

---

## Checklist de Implementación

### Archivos Nuevos

- [ ] `src/research/__init__.py`
- [ ] `src/research/morning_agent.py`
- [ ] `src/research/integration.py`
- [ ] `src/research/data_fetchers/__init__.py`
- [ ] `src/research/data_fetchers/base.py`
- [ ] `src/research/data_fetchers/grok_fetcher.py`
- [ ] `src/research/data_fetchers/sec_fetcher.py`
- [ ] `src/research/data_fetchers/market_fetcher.py`
- [ ] `src/research/data_fetchers/earnings_fetcher.py`
- [ ] `src/research/data_fetchers/economic_fetcher.py`
- [ ] `src/research/data_fetchers/news_fetcher.py`
- [ ] `src/research/prompts/__init__.py`
- [ ] `src/research/prompts/system_prompt.py`
- [ ] `src/research/prompts/context_template.py`
- [ ] `src/research/prompts/task_template.py`
- [ ] `src/research/models/__init__.py`
- [ ] `src/research/models/trading_idea.py`
- [ ] `src/research/models/daily_brief.py`
- [ ] `src/research/models/market_context.py`
- [ ] `src/dashboard/pages/5_Research.py`

### Archivos Modificados

- [ ] `src/models/social_message.py` - Añadir `SourceType.RESEARCH`
- [ ] `src/config/settings.py` - Añadir `ResearchConfig`
- [ ] `config/settings.yaml` - Añadir sección `research`

### Tests

- [ ] `tests/research/test_morning_agent.py`
- [ ] `tests/research/test_integration.py`
- [ ] `tests/research/data_fetchers/test_*.py`
- [ ] `tests/research/models/test_*.py`

---

## Próximos Pasos

1. Implementar modelos de datos (`TradingIdea`, `DailyBrief`)
2. Implementar Data Fetchers (empezando por `market_fetcher` y `grok_fetcher`)
3. Implementar prompts y `MorningResearchAgent`
4. Implementar integración con sistema existente
5. Implementar página de Dashboard
6. Añadir configuración a settings
7. Testing completo
8. Deploy y monitoreo

---

*Diseño aprobado: 2026-01-18*
