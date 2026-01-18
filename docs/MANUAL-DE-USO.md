# Manual de Uso - Sistema de Trading Intradiario

## Índice

1. [Descripción General](#1-descripción-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Instalación](#3-instalación)
4. [Configuración](#4-configuración)
5. [Ejecución](#5-ejecución)
6. [Componentes del Sistema](#6-componentes-del-sistema)
7. [Dashboard](#7-dashboard)
8. [Alertas y Notificaciones](#8-alertas-y-notificaciones)
9. [Journal de Trading](#9-journal-de-trading)
10. [Gestión de Riesgo](#10-gestión-de-riesgo)
11. [Solución de Problemas](#11-solución-de-problemas)

---

## 1. Descripción General

Sistema de trading intradiario que combina:
- **Morning Research Agent** - Daily Briefs pre-mercado con ideas de trading
- **Señales sociales via Grok** (X/Twitter con sentiment nativo)
- **Análisis profundo con Claude** (catalizadores, riesgo, contexto)
- **Validación técnica** (RSI, MACD, ADX, volumen)
- **Gestión de riesgo** (circuit breakers, position sizing)
- **Ejecución automatizada** via Alpaca API

### Filosofía: Research + Signal Flow

El Morning Research Agent genera ideas pre-mercado. Durante el día, Grok detecta señales sociales que se validan técnicamente antes de ejecutar.

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                MORNING RESEARCH AGENT                       │
│   (Pre-market: 12:00 y 15:00 Madrid)                        │
│   Futures + VIX + Gappers → Claude → Daily Brief + Ideas    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    MARKET GATE (Capa 0)                     │
│   Horario OK? │ Volumen OK? │ VIX OK? │ No choppy?          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   COLLECTORS (Capa 1)                       │
│   Grok (X/Twitter con sentiment nativo)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ANALYZERS (Capa 2)                        │
│   Claude AI (catalizadores, riesgo, contexto)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                TECHNICAL VALIDATION (Capa 3)                │
│   RSI │ MACD │ ADX │ Volume │ Options Flow                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  SCORING ENGINE (Capa 4)                    │
│   Score = Sentiment(50%) + Technical(50%) + Bonuses         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 RISK MANAGEMENT (Capa 5)                    │
│   Circuit Breakers │ Position Sizing │ Daily Limits         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION (Capa 6)                       │
│   Alpaca API → Paper/Live │ Journal Logging                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Instalación

### Requisitos

- Python 3.12+
- uv (gestor de paquetes)

### Pasos

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd intraday-trading-system

# 2. Instalar dependencias
uv sync

# 3. Copiar configuración de ejemplo
cp .env.example .env

# 4. Editar .env con tus API keys
nano .env

# 5. Crear directorios de datos
mkdir -p data/trades data/signals data/cache

# 6. Verificar instalación
uv run pytest -x -q
```

---

## 4. Configuración

### 4.1 Variables de Entorno (.env)

```bash
# Grok/xAI (REQUERIDO para señales sociales)
XAI_API_KEY=xai-XXXXXXXXXXXXXXXX

# Alpaca (REQUERIDO)
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_PAPER=true

# Claude AI (REQUERIDO para análisis y Research Agent)
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXX

# Telegram (REQUERIDO para alertas y Daily Briefs)
TELEGRAM_BOT_TOKEN=1234567890:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TELEGRAM_CHAT_ID=123456789
```

### 4.2 Archivo de Configuración (config/settings.yaml)

#### Secciones Principales:

**research:** Morning Research Agent
```yaml
research:
  enabled: true
  timezone: "Europe/Madrid"
  initial_brief_time: "12:00"     # Daily Brief inicial
  pre_open_brief_time: "15:00"    # Brief pre-apertura
  claude_model: "claude-sonnet-4-20250514"
  max_ideas: 5                    # Ideas por brief
  telegram_enabled: true
```

**system:** Configuración general
```yaml
system:
  name: "Intraday Trading System"
  mode: "paper"  # paper | live
  timezone: "America/New_York"
```

**collectors:** Recolección de señales
```yaml
collectors:
  grok:
    enabled: true
    refresh_interval_seconds: 30
    search_queries:
      - "$NVDA"
      - "$TSLA"
      - "unusual options activity"
      - "from:unusual_whales"
```

**market_gate:** Condiciones de mercado
```yaml
market_gate:
  trading_start: "09:30"
  trading_end: "16:00"
  avoid_lunch: true
  vix_max: 30.0
```

**scoring:** Sistema de puntuación
```yaml
scoring:
  tier_strong_threshold: 80    # Score >= 80: señal fuerte
  tier_moderate_threshold: 60  # Score >= 60: señal moderada
  tier_weak_threshold: 40      # Score >= 40: señal débil
```

**risk:** Gestión de riesgo
```yaml
risk:
  circuit_breakers:
    per_trade:
      max_loss_percent: 1.0    # Máx 1% pérdida por trade
    daily:
      max_loss_percent: 3.0    # Máx 3% pérdida diaria
```

---

## 5. Ejecución

### 5.1 Iniciar Sistema Principal

```bash
uv run python main.py
```

### 5.2 Iniciar Dashboard

```bash
uv run streamlit run src/dashboard/Home.py --server.port 8501
```

### 5.3 Ejecutar Tests

```bash
# Todos los tests
uv run pytest

# Solo tests de integración
uv run pytest tests/integration/

# Tests de validación
uv run pytest tests/validation/

# Con cobertura
uv run pytest --cov=src --cov-report=html
```

---

## 6. Componentes del Sistema

### 6.0 Morning Research Agent (src/research/)

Genera Daily Briefs con ideas de trading antes de la apertura.

```python
from src.research import MorningResearchAgent
from src.config.settings import Settings

settings = Settings.from_yaml("config/settings.yaml")
agent = MorningResearchAgent(settings.research)

# Generar brief inicial (12:00 Madrid)
brief = await agent.generate_brief("initial")

# brief.market_context = "Futures +0.5%, VIX 15.2"
# brief.ideas = [TradingIdea(...), ...]
# brief.watchlist = ["NVDA", "TSLA", "AMD"]
# brief.key_events = ["NVDA earnings after close"]

# Enviar por Telegram
await agent.send_brief(brief)
```

**Tipos de Brief:**
- `initial` - A las 12:00 Madrid: análisis overnight, futures, calendario
- `pre_open` - A las 15:00 Madrid: gappers, flow, ideas finales

**Ideas de Trading:**
Cada idea incluye:
- Símbolo y dirección (bullish/bearish)
- Entry, stop loss, target
- Conviction (high/medium/low)
- Thesis (razón para el trade)

### 6.1 Collectors (src/collectors/)

Recolectan señales de X/Twitter via Grok API.

```python
from src.collectors import GrokCollector

# Crear collector
grok = GrokCollector(
    api_key="xai-...",
    search_queries=["$NVDA", "$TSLA", "unusual options activity"],
    refresh_interval=30,
)

# Conectar e iniciar stream
await grok.connect()

async for message in grok.stream():
    print(f"Signal from @{message.author}: {message.content}")
```

### 6.2 Analyzers (src/analyzers/)

Analizan catalizadores y riesgo con Claude.

```python
from src.analyzers import ClaudeAnalyzer

# Análisis con Claude
claude = ClaudeAnalyzer()
analysis = await claude.analyze_catalyst(message)

# analysis.catalyst_type = "earnings_beat"
# analysis.risk_factors = ["high_iv", "market_uncertainty"]
# analysis.sentiment_direction = "bullish"
# analysis.confidence = 0.85
```

**Nota:** El sentiment nativo viene incluido en las respuestas de Grok.

### 6.3 Technical Validator (src/validators/)

Valida señales con indicadores técnicos.

```python
from src.validators import TechnicalValidator

validator = TechnicalValidator(alpaca_client)
result = await validator.validate(analyzed_message)

# result.status = "pass" | "warn" | "veto"
# result.rsi, result.macd_histogram, result.adx_value
```

### 6.4 Scoring Engine (src/scoring/)

Calcula score final y genera recomendaciones.

```python
from src.scoring import SignalScorer

scorer = SignalScorer(
    credibility_manager=credibility_manager,
    time_calculator=time_calculator,
    confluence_detector=confluence_detector,
    weight_calculator=weight_calculator,
    recommendation_builder=recommendation_builder,
)
recommendation = scorer.score(
    validated_signal=validated_signal,
    current_price=140.75,
)

# recommendation.final_score = 85
# recommendation.tier = "strong"
# recommendation.position_size_percent = 100
```

### 6.5 Market Gate (src/gate/)

Verifica condiciones de mercado.

```python
from src.gate import MarketGate

gate = MarketGate(alpaca_client, settings)
status = await gate.check()

# status.is_open = True | False
# status.reason = "Market open, conditions favorable"
# status.vix_level, status.spy_volume
```

### 6.6 Risk Manager (src/risk/)

Gestiona riesgo y circuit breakers.

```python
from src.risk import RiskManager

risk = RiskManager(settings)
decision = risk.evaluate_trade(recommendation)

# decision.approved = True | False
# decision.reason = "Within daily limits"
# decision.position_size = 500.0
```

### 6.7 Execution Manager (src/execution/)

Ejecuta trades via Alpaca.

```python
from src.execution import TradeExecutor

executor = TradeExecutor(alpaca_client, risk_manager)
result = await executor.execute(
    recommendation=recommendation,
    risk_result=risk_result,
    gate_status=gate_status,
)

# result.success, result.order_id, result.filled_price
```

### 6.8 Journal Manager (src/journal/)

Registra y analiza trades.

```python
from src.journal import JournalManager

journal = JournalManager(data_dir="data/trades")

# Registrar entrada
await journal.log_entry(trade_data)

# Obtener métricas
metrics = journal.calculate_metrics(period_days=30)
# metrics.win_rate, metrics.profit_factor, metrics.expectancy
```

---

## 7. Dashboard

### 7.1 Acceso

```
http://localhost:8501
```

### 7.2 Páginas Disponibles

1. **Home** - Vista general del sistema y estado
2. **Monitoreo** - Monitoreo en tiempo real (posiciones, señales, circuit breakers)
3. **Análisis** - Análisis de rendimiento (métricas, gráficos, patrones)
4. **Control** - Control del sistema (parámetros de riesgo, gate)
5. **Research** - Daily Briefs, ideas de trading, historial

### 7.3 Página de Research

En la página Research puedes:
- Ver Daily Briefs anteriores
- Explorar ideas de trading individuales
- Filtrar por fecha, tipo de brief, o símbolo
- Ver métricas de las ideas (si fueron ejecutadas)

### 7.4 Alertas en Dashboard

- 🟢 **Verde**: Sistema funcionando normal
- 🟡 **Amarillo**: Advertencia (VIX elevado, etc.)
- 🔴 **Rojo**: Circuit breaker activado

---

## 8. Alertas y Notificaciones

### 8.1 Tipos de Alertas (Telegram)

| Tipo | Descripción |
|------|-------------|
| `daily_brief` | Daily Brief del Morning Research Agent |
| `new_signal` | Nueva señal detectada via Grok |
| `entry_executed` | Trade ejecutado |
| `exit_executed` | Posición cerrada |
| `circuit_breaker` | Límite de riesgo alcanzado |
| `daily_summary` | Resumen diario |

### 8.2 Formato de Alerta de Señal

```
🎯 SEÑAL: $NVDA
Score: 85/100 ⭐⭐⭐⭐

📊 SOCIAL
• @unusual_whales: Large call sweep
• Sentiment: 0.89 bullish

📈 TÉCNICO
• RSI: 58 (neutral)
• MACD: bullish crossover
• ADX: 32 (trending)

💰 PLAN
• Entry: $140.75
• Stop: $139.20 (-1.1%)
• Target: $145.40 (+3.3%)
• Size: 322 shares

[EJECUTAR] [SKIP]
```

### 8.3 Daily Brief (Morning Research Agent)

Recibes dos briefs automáticos:

**Brief Inicial (12:00 Madrid / 6:00 NY):**
```
📊 DAILY BRIEF - INITIAL

🌍 OVERNIGHT
• Futures: ES +0.3%, NQ +0.5%
• VIX: 15.2 (-0.8)
• Asia: Nikkei +1.2%, HSI +0.4%

💡 IDEAS (3)
1. NVDA 🟢 | Entry: $450 | Target: $465 | Stop: $442
   Thesis: AI chip demand, data center growth

2. TSLA 🔴 | Entry: $180 | Target: $168 | Stop: $185
   Thesis: Delivery miss concerns, competition

📅 CALENDAR
• 8:30 AM: Initial Claims
• NVDA earnings after close
```

**Brief Pre-Open (15:00 Madrid / 9:00 NY):**
```
📊 DAILY BRIEF - PRE-OPEN

🔥 GAPPERS
• AMD +3.2% (upgrade)
• COIN -2.1% (SEC news)

📈 OPTIONS FLOW
• NVDA: Large call sweep 450c
• TSLA: Put buying 175p

💡 FINAL IDEAS
[Ideas actualizadas con data pre-market]
```

---

## 9. Journal de Trading

### 9.1 Datos Capturados Automáticamente

- Fecha/hora de entrada y salida
- Símbolo y dirección (long/short)
- Precio de entrada y salida
- Tamaño de posición
- P&L realizado
- Score de la señal
- Fuente del trigger
- Condiciones de mercado

### 9.2 Métricas Calculadas

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| Win Rate | % de trades ganadores | > 50% |
| Profit Factor | Ganancias / Pérdidas | > 1.5 |
| Expectancy | Ganancia promedio por trade | > 0.5R |
| Avg Win/Loss | Ratio ganancia/pérdida promedio | > 1.5 |
| Max Drawdown | Máxima caída desde peak | < 15% |
| Sharpe Ratio | Retorno ajustado por riesgo | > 1.0 |

### 9.3 Exportar Datos

```python
journal = JournalManager()

# Obtener trades como DataFrame
df = journal.get_trades_dataframe(period_days=30)

# Exportar a CSV
df.to_csv("trades_export.csv")
```

---

## 10. Gestión de Riesgo

### 10.1 Circuit Breakers

| Nivel | Límite | Acción |
|-------|--------|--------|
| Per-Trade | 1% pérdida | Stop loss automático |
| Diario | 3% pérdida | Detener trading, cooldown 60min |
| Semanal | 6% pérdida | Forzar modo paper |

### 10.2 Position Sizing

Basado en riesgo fijo:
```
Position Size = Risk Amount / Stop Distance

Ejemplo:
- Capital: $50,000
- Riesgo por trade: 1% = $500
- Stop distance: 1.5%
- Position Size: $500 / 1.5% = $33,333
```

### 10.3 Niveles de Riesgo

| Nivel | Riesgo/Trade | Max Posición | Posiciones |
|-------|--------------|--------------|------------|
| Conservative | 0.5% | 2.5% | 2 |
| Standard | 1.0% | 5.0% | 3 |
| Aggressive | 1.5% | 7.5% | 4 |

### 10.4 Behavioral Detection

El sistema detecta y bloquea:
- **Revenge Trading**: Pérdida seguida de posición mayor
- **Overtrading**: > 3 trades/hora o > 10/día
- **FOMO**: Entry después de move > 5%
- **Stop Widening**: Modificación de stops

---

## 11. Solución de Problemas

### 11.1 Error de Conexión Alpaca

```
Error: Could not connect to Alpaca API
```

**Solución:**
1. Verificar API keys en `.env`
2. Verificar que `ALPACA_PAPER=true` para paper trading
3. Verificar conexión a internet

### 11.2 Error de Telegram

```
Error: Telegram bot token invalid
```

**Solución:**
1. Verificar token con BotFather
2. Asegurar que el bot está iniciado (`/start`)
3. Verificar `TELEGRAM_CHAT_ID`

### 11.3 Error de Grok/xAI API

```
Error: XAI_API_KEY not found
```

**Solución:**
1. Verificar que `.env` tiene `XAI_API_KEY=xai-...`
2. Confirmar el formato correcto de la key
3. Verificar créditos en [xAI Console](https://console.x.ai/)

### 11.4 Error de Claude API

```
Error: Anthropic API authentication failed
```

**Solución:**
1. Verificar `ANTHROPIC_API_KEY`
2. Verificar que la cuenta tiene créditos
3. Sin Claude, el Morning Research Agent no funcionará

### 11.5 Market Gate Siempre Cerrado

```
Gate status: CLOSED - Outside trading hours
```

**Verificar:**
1. Horario del sistema (timezone: America/New_York)
2. Que sea día de mercado (no fin de semana/feriado)
3. Configuración `trading_start` y `trading_end`

### 11.6 Sin Señales Generadas

**Posibles causas:**
1. GrokCollector no conectado
2. Score threshold muy alto
3. Gate cerrado
4. Circuit breaker activado

**Debug:**
```python
# Verificar estado del sistema
print(f"Gate: {gate.check()}")
print(f"Risk: {risk.get_status()}")
print(f"Messages buffered: {orchestrator.buffer_size}")
```

### 11.7 Daily Briefs No Aparecen

**Verificar:**
1. `research.enabled: true` en settings.yaml
2. `ANTHROPIC_API_KEY` configurado correctamente
3. Horario correcto (12:00 y 15:00 hora Madrid)
4. Directorio `data/research/briefs` existe

### 11.8 Tests Fallando

```bash
# Verificar dependencias
uv sync

# Ejecutar test específico con verbose
uv run pytest tests/path/test_file.py -v -s
```

---

## Apéndice A: Comandos Rápidos

```bash
# Iniciar sistema
uv run python main.py

# Iniciar dashboard
uv run streamlit run src/dashboard/Home.py

# Ejecutar tests
uv run pytest

# Ver logs
tail -f data/logs/trading.log

# Verificar estado de cuenta
uv run python -c "
from src.execution import AlpacaClient
import asyncio

async def check():
    c = AlpacaClient(paper=True)
    await c.connect()
    print(await c.get_account())
    await c.disconnect()

asyncio.run(check())
"
```

---

## Apéndice B: Glosario

| Término | Definición |
|---------|------------|
| **ATR** | Average True Range - volatilidad |
| **Circuit Breaker** | Límite de pérdida que detiene trading |
| **Confluence** | Múltiples señales apuntando misma dirección |
| **Gate** | Filtro de condiciones de mercado |
| **R-Multiple** | Ganancia/Pérdida en unidades de riesgo |
| **RVOL** | Relative Volume - volumen vs promedio |
| **Sentiment** | Análisis de sentimiento de texto |
| **VIX** | Índice de volatilidad del mercado |
| **VWAP** | Volume Weighted Average Price |

---

*Manual actualizado: 2026-01-18*
*Versión: 2.0.0 (Morning Research Agent integrado)*
