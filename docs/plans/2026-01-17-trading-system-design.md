# Sistema de Trading Intradiario con IA y Análisis Social

**Fecha:** 2026-01-17
**Estado:** Diseño Validado
**Versión:** 1.0

---

## 1. Resumen Ejecutivo

Sistema de trading intradiario que combina análisis de redes sociales (X/Twitter, Reddit, Stocktwits) con validación técnica y ejecución automatizada a través de Alpaca API. El enfoque es **social-first**: detectar oportunidades en redes sociales primero, luego validar técnicamente antes de ejecutar.

### Objetivos Principales
- Detección temprana de catalizadores en redes sociales
- Análisis de sentimiento con FinTwitBERT + Claude API
- Validación técnica multi-timeframe
- Gestión de riesgo profesional con circuit breakers
- Alertas en tiempo real via Telegram + Dashboard

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPA 0: MARKET CONDITION GATE                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Horario OK? │ │ Volumen OK? │ │ VIX < 30?   │ │ No choppy market?   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                    ↓ Si TODOS pasan, continúa                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CAPA 1: COLECCIÓN DE DATOS                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ X/Twitter    │ │ Reddit       │ │ Stocktwits   │ │ Alpaca News  │       │
│  │ (twscrape)   │ │ (asyncpraw)  │ │ (pytwits)    │ │ (alpaca-py)  │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CAPA 2: PROCESAMIENTO Y ANÁLISIS                      │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐    │
│  │ FinTwitBERT      │ │ Extractor de     │ │ Claude API               │    │
│  │ Sentiment Score  │ │ Tickers ($AAPL)  │ │ Deep Analysis            │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CAPA 3: VALIDACIÓN TÉCNICA                            │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐    │
│  │ Multi-Timeframe  │ │ VWAP/ORB/ATR     │ │ Volume Confirmation      │    │
│  │ 15m → 5m → 1m    │ │ Technical Setup  │ │ Relative Volume > 2x     │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CAPA 4: MOTOR DE SCORING                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Score Final = Social(40%) + Technical(35%) + Context(25%)            │  │
│  │ Umbral mínimo: 70/100 para considerar operación                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CAPA 5: CIRCUIT BREAKERS (Triple Capa)                   │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐    │
│  │ Per-Trade: 1%    │ │ Daily: 3%        │ │ Weekly: 6%               │    │
│  │ Max loss/trade   │ │ Max loss/day     │ │ Max loss/week            │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Behavioral Detection: Revenge trading, Overtrading, FOMO, Stop widening│
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAPA 6: EJECUCIÓN                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Alpaca API → Paper Trading (Nivel 1-2) → Live Trading (Nivel 3)      │  │
│  │ Auto-journal + Métricas + Walk-forward validation                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dependencias y Librerías

### 3.1 Dependencias Directas (pip install)

| Librería | Propósito | Documentación |
|----------|-----------|---------------|
| `alpaca-py` | SDK oficial Alpaca (trading, news, screener) | [GitHub](https://github.com/alpacahq/alpaca-py) |
| `asyncpraw` | Reddit API async para streaming en tiempo real | [Docs](https://asyncpraw.readthedocs.io/) |
| `twscrape` | Twitter scraper con pool de cuentas | [GitHub](https://github.com/vladkens/twscrape) |
| `twikit` | Twitter scraper sin API oficial | [GitHub](https://github.com/d60/twikit) |
| `pytwits` | Wrapper de Stocktwits API | [GitHub](https://github.com/khmurakami/PyTwits) |
| `transformers` | Para cargar FinTwitBERT | [HuggingFace](https://huggingface.co/) |
| `vectorbt` | Backtesting y walk-forward validation | [Docs](https://vectorbt.dev/) |
| `streamlit` | Dashboard interactivo | [Docs](https://streamlit.io/) |
| `anthropic` | Claude API para análisis profundo | [Docs](https://docs.anthropic.com/) |
| `python-telegram-bot` | Alertas y checklist interactivo | [Docs](https://python-telegram-bot.org/) |

### 3.2 Arquitecturas de Referencia (patrones a adaptar)

| Repositorio | Qué Adaptar |
|-------------|-------------|
| [FinTwit-Bot](https://github.com/StephanAkkerman/fintwit-bot) | Agregación multi-fuente, integración FinTwitBERT |
| [Tarzan](https://github.com/greenmachine112/tarzan) | Estrategia inversa de sentimiento, weighted scoring |
| [nlp-sentiment-quant-monitor](https://github.com/Laurenz-Thuemmler/nlp-sentiment-quant-monitor) | Pipeline FinBERT, procesamiento batch |
| [jnech1997/day-trader](https://github.com/jnech1997/day-trader) | Indicadores VWAP/ATR/RSI |
| [Reddit-Stock-Sentiment-Analyzer](https://github.com/Adith-Rai/Reddit-Stock-Sentiment-Analyzer) | LLM batching para Reddit |

---

## 4. Estructura del Proyecto

```
intraday-trading-system/
├── config/
│   ├── settings.yaml              # Configuración principal
│   ├── risk_params.yaml           # Parámetros de riesgo por nivel
│   └── social_sources.yaml        # Cuentas Twitter, subreddits, filtros
│
├── src/
│   ├── __init__.py
│   │
│   ├── gate/                      # CAPA 0: Market Condition Gate
│   │   ├── __init__.py
│   │   ├── market_hours.py        # Verificar horario de trading
│   │   ├── volume_check.py        # Verificar volumen del mercado
│   │   ├── volatility_check.py    # VIX y condiciones de mercado
│   │   └── gate_manager.py        # Orquestador de la puerta
│   │
│   ├── collectors/                # CAPA 1: Colección de datos
│   │   ├── __init__.py
│   │   ├── twitter_collector.py   # twscrape + twikit
│   │   ├── reddit_collector.py    # asyncpraw streaming
│   │   ├── stocktwits_collector.py # pytwits
│   │   ├── news_collector.py      # Alpaca News API
│   │   └── collector_manager.py   # Orquestador de collectors
│   │
│   ├── analyzers/                 # CAPA 2: Procesamiento
│   │   ├── __init__.py
│   │   ├── ticker_extractor.py    # Extraer $TICKER de texto
│   │   ├── sentiment_analyzer.py  # FinTwitBERT sentiment
│   │   ├── claude_analyzer.py     # Claude API deep analysis
│   │   └── analyzer_manager.py    # Pipeline de análisis
│   │
│   ├── technical/                 # CAPA 3: Validación técnica
│   │   ├── __init__.py
│   │   ├── indicators.py          # VWAP, ATR, RSI, ORB
│   │   ├── multi_timeframe.py     # Análisis 15m/5m/1m
│   │   ├── volume_profile.py      # Relative volume, RVOL
│   │   └── technical_validator.py # Validador consolidado
│   │
│   ├── engine/                    # CAPA 4: Motor de scoring
│   │   ├── __init__.py
│   │   ├── scoring_engine.py      # Cálculo de score final
│   │   ├── opportunity_ranker.py  # Ranking de oportunidades
│   │   └── signal_generator.py    # Generación de señales
│   │
│   ├── risk/                      # CAPA 5: Gestión de riesgo
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py     # Triple capa de protección
│   │   ├── position_sizer.py      # Cálculo de tamaño de posición
│   │   ├── behavioral_detector.py # Detección de patrones negativos
│   │   └── risk_manager.py        # Orquestador de riesgo
│   │
│   ├── execution/                 # CAPA 6: Ejecución
│   │   ├── __init__.py
│   │   ├── alpaca_client.py       # Cliente Alpaca unificado
│   │   ├── order_manager.py       # Gestión de órdenes
│   │   ├── position_tracker.py    # Tracking de posiciones
│   │   └── execution_engine.py    # Motor de ejecución
│   │
│   ├── notifications/             # Alertas
│   │   ├── __init__.py
│   │   ├── telegram_bot.py        # Bot de Telegram
│   │   ├── alert_formatter.py     # Formato de alertas
│   │   └── checklist_handler.py   # Pre-market checklist
│   │
│   ├── journal/                   # Trading Journal
│   │   ├── __init__.py
│   │   ├── trade_logger.py        # Logging automático
│   │   ├── metrics_calculator.py  # Win rate, profit factor, etc.
│   │   ├── pattern_analyzer.py    # Análisis de patrones propios
│   │   └── journal_manager.py     # Orquestador del journal
│   │
│   ├── dashboard/                 # Dashboard Streamlit
│   │   ├── __init__.py
│   │   ├── app.py                 # Aplicación principal
│   │   ├── pages/
│   │   │   ├── live_signals.py    # Señales en tiempo real
│   │   │   ├── positions.py       # Posiciones actuales
│   │   │   ├── journal.py         # Trading journal
│   │   │   └── analytics.py       # Métricas y análisis
│   │   └── components/
│   │       ├── signal_card.py     # Tarjeta de señal
│   │       └── metrics_panel.py   # Panel de métricas
│   │
│   └── validation/                # Backtesting
│       ├── __init__.py
│       ├── backtester.py          # VectorBT backtesting
│       ├── walk_forward.py        # Walk-forward validation
│       └── level_promoter.py      # Promoción entre niveles
│
├── data/
│   ├── trades/                    # Historial de trades
│   ├── signals/                   # Señales generadas
│   ├── backtest_results/          # Resultados de backtest
│   └── cache/                     # Cache de datos
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
│   └── plans/
│       └── 2026-01-17-trading-system-design.md
│
├── scripts/
│   ├── setup_twitter_accounts.py  # Configurar pool de cuentas
│   └── run_backtest.py            # Ejecutar backtests
│
├── main.py                        # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Configuración

### 5.1 settings.yaml

```yaml
# config/settings.yaml
# Configuración principal del sistema de trading

system:
  name: "Intraday Trading System"
  version: "1.0.0"
  mode: "paper"  # paper | live
  timezone: "America/New_York"

# CAPA 0: Market Condition Gate
market_gate:
  enabled: true
  trading_hours:
    start: "09:30"
    end: "16:00"
    avoid_lunch: true
    lunch_start: "11:30"
    lunch_end: "14:00"

  volume_requirements:
    spy_min_volume_1m: 500000
    qqq_min_volume_1m: 300000

  volatility_limits:
    vix_max: 30
    vix_elevated: 25
    reduce_size_when_elevated: true
    size_reduction_factor: 0.5

  choppy_market_detection:
    enabled: true
    atr_threshold_multiplier: 0.3
    range_vs_atr_ratio: 1.5

# CAPA 1: Colección de datos
collectors:
  twitter:
    enabled: true
    engine: "twscrape"  # twscrape | twikit
    accounts_pool_size: 5
    rate_limit_buffer: 0.8
    refresh_interval_seconds: 15

  reddit:
    enabled: true
    use_streaming: true
    batch_fallback_interval: 60

  stocktwits:
    enabled: true
    refresh_interval_seconds: 30

  alpaca_news:
    enabled: true
    use_streaming: true

# CAPA 2: Análisis
analyzers:
  sentiment:
    model: "StephanAkkerman/FinTwitBERT-sentiment"
    batch_size: 32
    min_confidence: 0.7

  claude:
    enabled: true
    model: "claude-sonnet-4-20250514"
    max_tokens: 1000
    use_for:
      - catalyst_classification
      - risk_assessment
      - context_analysis
    rate_limit_per_minute: 20

  ticker_extraction:
    min_market_cap: 100000000  # $100M minimum
    exclude_crypto: true
    exclude_otc: true

# CAPA 3: Técnico
technical:
  multi_timeframe:
    enabled: true
    timeframes: ["15m", "5m", "1m"]
    require_alignment: true

  indicators:
    vwap:
      enabled: true
      bands: [1, 2]
    atr:
      period: 14
    rsi:
      period: 14
      overbought: 70
      oversold: 30
    orb:
      period_minutes: 15

  volume:
    relative_volume_min: 2.0
    average_period_days: 20

# CAPA 4: Scoring
scoring:
  weights:
    social: 0.40
    technical: 0.35
    context: 0.25

  thresholds:
    minimum_to_consider: 70
    strong_signal: 85
    exceptional: 95

  social_score_components:
    sentiment_weight: 0.35
    source_quality_weight: 0.30
    velocity_weight: 0.20
    consensus_weight: 0.15

# CAPA 5: Risk Management
risk:
  circuit_breakers:
    per_trade:
      max_loss_percent: 1.0
      hard_stop: true
    daily:
      max_loss_percent: 3.0
      max_trades_after_loss: 0
      cooldown_minutes: 60
    weekly:
      max_loss_percent: 6.0
      force_paper_mode: true

  position_sizing:
    method: "percent_risk"  # percent_risk | fixed | kelly
    max_position_percent: 5.0
    max_positions: 3

  behavioral_detection:
    enabled: true
    patterns:
      revenge_trading:
        lookback_minutes: 30
        loss_then_larger_size: true
      overtrading:
        max_trades_per_hour: 3
        max_trades_per_day: 10
      fomo:
        entry_after_big_move_percent: 5.0
      stop_widening:
        detect_modifications: true

# CAPA 6: Ejecución
execution:
  alpaca:
    paper_url: "https://paper-api.alpaca.markets"
    live_url: "https://api.alpaca.markets"
    order_type: "limit"  # limit | market
    limit_offset_percent: 0.05
    time_in_force: "day"

  partial_exits:
    enabled: true
    rules:
      - at_r_multiple: 1.0
        exit_percent: 33
        move_stop_to: "breakeven"
      - at_r_multiple: 2.0
        exit_percent: 33
        trail_stop_percent: 1.0
      - at_r_multiple: 3.0
        exit_percent: 100

# Notifications
notifications:
  telegram:
    enabled: true
    alert_types:
      - new_signal
      - entry_executed
      - exit_executed
      - circuit_breaker_triggered
      - daily_summary

    pre_market_checklist:
      enabled: true
      time: "09:00"
      items:
        - "Economic calendar reviewed"
        - "Overnight news checked"
        - "Watchlist prepared"
        - "Mental state: focused"
        - "Risk parameters confirmed"

  dashboard:
    enabled: true
    port: 8501
    refresh_interval_seconds: 5

# Journal
journal:
  auto_logging: true
  capture:
    - entry_reason
    - exit_reason
    - emotions_tag
    - market_conditions
    - screenshots

  metrics:
    calculate:
      - win_rate
      - profit_factor
      - expectancy
      - avg_win_loss_ratio
      - max_drawdown
      - sharpe_ratio
    period: "rolling_30_days"

  review:
    weekly_report: true
    report_day: "saturday"

# Validation
validation:
  levels:
    - name: "Level 1 - Backtest"
      type: "backtest"
      min_trades: 100
      min_profit_factor: 1.3
      max_drawdown: 15

    - name: "Level 2 - Walk-Forward"
      type: "walk_forward"
      in_sample_months: 3
      out_of_sample_months: 1
      min_oos_profit_factor: 1.2

    - name: "Level 3 - Paper"
      type: "paper"
      min_trades: 50
      min_profit_factor: 1.2
      duration_weeks: 4

    - name: "Level 4 - Live Small"
      type: "live"
      position_size_multiplier: 0.25
      duration_weeks: 4

    - name: "Level 5 - Live Full"
      type: "live"
      position_size_multiplier: 1.0
```

### 5.2 risk_params.yaml

```yaml
# config/risk_params.yaml
# Parámetros de riesgo detallados por nivel

levels:
  level_1:
    name: "Conservative"
    description: "Para estrategias nuevas o después de drawdown"

    position:
      max_risk_per_trade: 0.5  # 0.5% del capital
      max_position_size: 2.5   # 2.5% del capital
      max_concurrent_positions: 2

    targets:
      min_r_multiple: 2.0
      take_profit_r: 3.0

    filters:
      min_score: 80
      min_relative_volume: 3.0
      require_multi_source_confirmation: true

    promotion:
      min_trades: 30
      min_win_rate: 0.55
      min_profit_factor: 1.5
      max_consecutive_losses: 3

  level_2:
    name: "Standard"
    description: "Operación normal"

    position:
      max_risk_per_trade: 1.0
      max_position_size: 5.0
      max_concurrent_positions: 3

    targets:
      min_r_multiple: 1.5
      take_profit_r: 2.5

    filters:
      min_score: 70
      min_relative_volume: 2.0
      require_multi_source_confirmation: false

    promotion:
      min_trades: 50
      min_win_rate: 0.50
      min_profit_factor: 1.3
      max_consecutive_losses: 4

    demotion:
      consecutive_losses: 5
      daily_loss_percent: 2.0
      weekly_loss_percent: 4.0

  level_3:
    name: "Aggressive"
    description: "Durante racha ganadora confirmada"

    position:
      max_risk_per_trade: 1.5
      max_position_size: 7.5
      max_concurrent_positions: 4

    targets:
      min_r_multiple: 1.0
      take_profit_r: 2.0

    filters:
      min_score: 65
      min_relative_volume: 1.5
      require_multi_source_confirmation: false

    demotion:
      consecutive_losses: 3
      daily_loss_percent: 1.5
      weekly_loss_percent: 3.0
      single_loss_over_percent: 2.0

partial_exit_rules:
  standard:
    - trigger: "1R"
      action: "exit_33%"
      stop_adjustment: "breakeven"
    - trigger: "2R"
      action: "exit_33%"
      stop_adjustment: "trail_1R"
    - trigger: "3R"
      action: "exit_remaining"

  conservative:
    - trigger: "1R"
      action: "exit_50%"
      stop_adjustment: "breakeven"
    - trigger: "2R"
      action: "exit_remaining"

stop_loss_rules:
  methods:
    atr_based:
      multiplier: 1.5
      min_distance_percent: 0.5
      max_distance_percent: 3.0

    technical:
      below_vwap: true
      below_support: true
      buffer_percent: 0.1

    time_based:
      max_hold_minutes: 120
      force_exit_at_close: true
      close_buffer_minutes: 15
```

### 5.3 social_sources.yaml

```yaml
# config/social_sources.yaml
# Fuentes de redes sociales a monitorear

twitter:
  smart_money:
    description: "Traders institucionales y profesionales"
    accounts:
      - username: "unusual_whales"
        weight: 1.0
        focus: ["options_flow", "dark_pool"]
      - username: "OptionsHawk"
        weight: 0.9
        focus: ["options_flow"]
      - username: "Fxhedgers"
        weight: 0.8
        focus: ["macro", "breaking_news"]
      - username: "zaborsky"
        weight: 0.9
        focus: ["technical", "momentum"]
      - username: "trikitrakes87"
        weight: 0.8
        focus: ["flow", "levels"]

  news_breaking:
    description: "Fuentes de noticias en tiempo real"
    accounts:
      - username: "FirstSquawk"
        weight: 1.0
        latency: "fastest"
      - username: "LiveSquawk"
        weight: 1.0
        latency: "fastest"
      - username: "DeItaone"
        weight: 0.9
        latency: "fast"
      - username: "Newsfilterio"
        weight: 0.8
        latency: "fast"
      - username: "financialjuice"
        weight: 0.8
        latency: "fast"

  cashtag_monitoring:
    enabled: true
    min_mentions_per_hour: 10
    exclude_crypto_cashtags: true
    focus_market_cap_min: 1000000000  # $1B+

reddit:
  tier_1_large:
    description: "Subreddits grandes con alta actividad"
    subreddits:
      - name: "wallstreetbets"
        weight: 0.7
        noise_level: "high"
        dd_weight: 1.2
      - name: "stocks"
        weight: 0.9
        noise_level: "medium"
      - name: "investing"
        weight: 0.8
        noise_level: "low"

  tier_2_sector:
    description: "Subreddits sectoriales"
    subreddits:
      - name: "semiconductor"
        weight: 1.0
        sector: "tech"
      - name: "biotech"
        weight: 1.0
        sector: "healthcare"
      - name: "energy_stocks"
        weight: 0.9
        sector: "energy"
      - name: "REITs"
        weight: 0.8
        sector: "real_estate"

  tier_3_dd_hunting:
    description: "Subreddits con DD de calidad"
    subreddits:
      - name: "ValueInvesting"
        weight: 1.2
        dd_quality: "high"
      - name: "SecurityAnalysis"
        weight: 1.3
        dd_quality: "highest"
      - name: "UndervaluedStonks"
        weight: 1.0
        dd_quality: "medium"

  filters:
    min_upvotes: 10
    min_comment_count: 5
    max_age_hours: 24
    exclude_meme_flairs: true
    require_ticker_mention: true

stocktwits:
  enabled: true
  refresh_interval: 30

  watchlist_mode: true  # Solo monitorear tickers en watchlist

  sentiment_threshold:
    bullish_min: 0.6
    bearish_max: 0.4

  message_velocity:
    spike_threshold_multiplier: 3.0
    lookback_hours: 1

source_quality_weights:
  smart_money_twitter: 1.0
  breaking_news: 0.95
  reddit_dd: 0.9
  reddit_general: 0.6
  stocktwits: 0.5
  cashtag_volume: 0.4

deduplication:
  enabled: true
  time_window_minutes: 5
  similarity_threshold: 0.8
```

---

## 6. Flujo Completo de Ejemplo

### Escenario: Detección de oportunidad en $NVDA

```
[08:45 AM] PRE-MARKET CHECKLIST via Telegram
┌─────────────────────────────────────────┐
│ 📋 Pre-Market Checklist                 │
│                                         │
│ ☐ Economic calendar reviewed            │
│ ☐ Overnight news checked                │
│ ☐ Watchlist prepared                    │
│ ☐ Mental state: focused                 │
│ ☐ Risk parameters confirmed             │
│                                         │
│ [Mark All Complete] [Skip Today]        │
└─────────────────────────────────────────┘
Usuario marca todos como completados ✓

[09:32 AM] MARKET GATE CHECK
✓ Horario OK (dentro de 9:30-16:00)
✓ Volumen OK (SPY 1m vol: 2.3M > 500K)
✓ VIX OK (18.5 < 30)
✓ No choppy market detectado
→ GATE OPEN: Permitido operar

[09:33 AM] TWITTER COLLECTOR detecta:
@unusual_whales: "🚨 Large $NVDA call sweep
$142 strike 2/21 exp, $2.4M premium,
bullish sentiment"

[09:33 AM] TICKER EXTRACTION
→ $NVDA extraído
→ Market cap: $3.2T ✓
→ No crypto ✓
→ No OTC ✓

[09:33 AM] SENTIMENT ANALYSIS (FinTwitBERT)
→ Score: 0.89 (muy bullish)
→ Confidence: 0.94

[09:33 AM] CLAUDE DEEP ANALYSIS
Request: "Analiza este flujo de opciones..."
Response: {
  "catalyst_type": "institutional_accumulation",
  "confidence": 0.85,
  "risk_factors": ["earnings_in_3_weeks"],
  "recommendation": "valid_catalyst",
  "reasoning": "Large sweep indicates conviction..."
}

[09:33 AM] TECHNICAL VALIDATION
15m: Tendencia alcista ✓ (above VWAP)
5m:  Setup válido ✓ (pullback to VWAP)
1m:  Entry zone ✓ (bouncing off VWAP)

VWAP: $140.50 (precio actual $140.75)
ATR(14): $2.30
RSI: 58 (neutral, espacio para subir)
RVOL: 2.8x (alto interés)

[09:33 AM] SCORING ENGINE
┌─────────────────────────────────────────┐
│ Social Score:    36/40 (90%)            │
│   - Sentiment:   12.6/14                │
│   - Source:      10.8/12 (unusual_whales)│
│   - Velocity:    7.2/8                  │
│   - Consensus:   5.4/6                  │
│                                         │
│ Technical Score: 31/35 (89%)            │
│   - Trend:       10/10                  │
│   - Setup:       9/10                   │
│   - Volume:      8/10                   │
│   - Indicators:  4/5                    │
│                                         │
│ Context Score:   21/25 (84%)            │
│   - Market cond: 8/10                   │
│   - Sector:      7/8                    │
│   - Timing:      6/7                    │
│                                         │
│ TOTAL SCORE:     88/100 ⭐              │
│ Threshold:       70 (PASSED)            │
└─────────────────────────────────────────┘

[09:33 AM] CIRCUIT BREAKER CHECK
✓ No pérdidas previas hoy
✓ Bajo límite semanal
✓ No behavioral patterns detectados
→ CLEARED FOR EXECUTION

[09:33 AM] POSITION SIZING
Capital: $50,000
Risk per trade: 1% = $500
Stop loss: $139.20 (below VWAP - ATR)
Distance: $1.55 (1.1%)
Position size: $500 / $1.55 = 322 shares
Position value: $45,361 (90% of limit OK)

[09:33 AM] TELEGRAM ALERT
┌─────────────────────────────────────────────────────┐
│ 🎯 SEÑAL: $NVDA                                     │
│ Score: 88/100 ⭐⭐⭐⭐                               │
│                                                     │
│ 📊 SOCIAL (36/40)                                   │
│ • @unusual_whales: Call sweep $2.4M                 │
│ • Sentiment: 0.89 bullish                           │
│ • Catalyst: institutional_accumulation              │
│                                                     │
│ 📈 TÉCNICO (31/35)                                  │
│ • 15m/5m/1m: Aligned bullish                        │
│ • Price: $140.75 (VWAP: $140.50)                    │
│ • RVOL: 2.8x                                        │
│                                                     │
│ 💰 PLAN DE TRADE                                    │
│ • Entry: $140.75 (limit)                            │
│ • Stop: $139.20 (-1.1%)                             │
│ • T1: $142.30 (+1.1%, 1R) → Exit 33%                │
│ • T2: $143.85 (+2.2%, 2R) → Exit 33%                │
│ • T3: $145.40 (+3.3%, 3R) → Exit 34%                │
│ • Size: 322 shares ($45,361)                        │
│ • Risk: $500 (1%)                                   │
│                                                     │
│ ⚠️ RISKS                                            │
│ • Earnings in 3 weeks                               │
│                                                     │
│ [EJECUTAR] [SKIP] [MODIFICAR]                       │
└─────────────────────────────────────────────────────┘

[09:33 AM] Usuario presiona [EJECUTAR]

[09:33 AM] ALPACA EXECUTION
Order submitted: BUY 322 NVDA @ $140.80 LIMIT
Order filled: 322 @ $140.76

[09:33 AM] JOURNAL AUTO-LOG
{
  "trade_id": "2026-01-17-NVDA-001",
  "entry_time": "09:33:45",
  "entry_price": 140.76,
  "shares": 322,
  "entry_reason": "unusual_whales_sweep_alert",
  "market_conditions": "bullish_trend_high_volume",
  "score": 88
}

[10:15 AM] FIRST TARGET HIT
Price reaches $142.30 (1R)
→ Auto-sell 106 shares @ $142.28
→ Stop moved to $140.76 (breakeven)
→ Telegram notification sent

[11:02 AM] SECOND TARGET HIT
Price reaches $143.85 (2R)
→ Auto-sell 106 shares @ $143.82
→ Trailing stop activated at $142.40

[11:45 AM] TRAILING STOP HIT
Price pulls back to $142.40
→ Auto-sell remaining 110 shares @ $142.38

[11:45 AM] TRADE SUMMARY
┌─────────────────────────────────────────┐
│ TRADE CLOSED: $NVDA                     │
│                                         │
│ Entry: $140.76                          │
│ Exits:                                  │
│   106 @ $142.28 (+$161)                 │
│   106 @ $143.82 (+$324)                 │
│   110 @ $142.38 (+$178)                 │
│                                         │
│ Total P&L: +$663 (+1.46%)               │
│ R-Multiple: 1.33R                       │
│                                         │
│ 📈 Daily Stats                          │
│ Trades: 1 | Win Rate: 100%              │
│ Daily P&L: +$663 (+1.33%)               │
└─────────────────────────────────────────┘
```

---

## 7. Mejoras Profesionales Integradas

### 7.1 Circuit Breakers (Triple Capa)

| Nivel | Límite | Acción |
|-------|--------|--------|
| Per-Trade | 1% max loss | Hard stop, no modificar |
| Diario | 3% max loss | Stop trading, 60min cooldown |
| Semanal | 6% max loss | Forzar modo paper |

### 7.2 Market Condition Gate

- **Horarios a evitar**: 11:30 AM - 2:00 PM (lunch hour, baja liquidez)
- **VIX monitoring**: Reducir tamaño 50% si VIX > 25
- **Choppy market**: Detectar rangos estrechos sin dirección

### 7.3 Multi-Timeframe Confirmation

- **15 minutos**: Dirección de tendencia principal
- **5 minutos**: Setup y estructura
- **1 minuto**: Timing de entrada preciso

Requiere alineación de los 3 timeframes para operar.

### 7.4 Auto Trading Journal

Captura automática de:
- Screenshots del setup
- Razón de entrada/salida
- Condiciones de mercado
- Tag de emociones (manual)
- Métricas calculadas en tiempo real

### 7.5 Pre-Market Checklist

Checklist interactivo en Telegram cada mañana:
- Calendario económico revisado
- Noticias overnight revisadas
- Watchlist preparada
- Estado mental: enfocado
- Parámetros de riesgo confirmados

### 7.6 Walk-Forward Validation

**Proceso de promoción entre niveles:**

```
Backtest (100+ trades, PF > 1.3)
    ↓
Walk-Forward (3m in-sample, 1m out-of-sample)
    ↓
Paper Trading (4 semanas, 50+ trades)
    ↓
Live Small (25% tamaño, 4 semanas)
    ↓
Live Full (100% tamaño)
```

### 7.7 Behavioral Pattern Detection

| Patrón | Detección | Acción |
|--------|-----------|--------|
| Revenge Trading | Pérdida seguida de posición mayor en <30min | Bloquear trade + alerta |
| Overtrading | >3 trades/hora o >10/día | Cooldown forzado |
| FOMO | Entry después de move >5% | Warning + confirmación extra |
| Stop Widening | Modificación de stop | Bloquear modificación |

---

## 8. Métricas de Éxito

### Métricas Primarias
- **Win Rate**: Objetivo > 50%
- **Profit Factor**: Objetivo > 1.5
- **Expectancy**: Objetivo > 0.5R por trade
- **Max Drawdown**: Límite < 15%

### Métricas Secundarias
- Trades por día (objetivo: 2-5)
- R-Multiple promedio
- Tiempo promedio en trade
- Tasa de circuit breaker activado

---

## 9. Timeline de Implementación

| Fase | Duración | Entregables |
|------|----------|-------------|
| **Fase 1: Core** | Semana 1-2 | Collectors, basic sentiment, Alpaca connection |
| **Fase 2: Analysis** | Semana 3-4 | FinTwitBERT, Claude integration, technical validation |
| **Fase 3: Risk** | Semana 5-6 | Circuit breakers, position sizing, behavioral detection |
| **Fase 4: Execution** | Semana 7-8 | Order management, partial exits, journal |
| **Fase 5: Interface** | Semana 9-10 | Telegram bot, Streamlit dashboard |
| **Fase 6: Validation** | Semana 11-12 | Backtesting, walk-forward, paper trading setup |

---

## 10. Notas de Seguridad

- **API Keys**: Almacenar en `.env`, nunca en código
- **Rate Limiting**: Respetar límites de todas las APIs
- **Paper First**: SIEMPRE empezar en paper trading
- **Circuit Breakers**: NUNCA desactivar en producción
- **Logs**: Mantener logs completos para auditoría

---

*Documento generado: 2026-01-17*
*Próximo paso: Implementación Fase 1*
