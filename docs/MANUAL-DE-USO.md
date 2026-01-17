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
- **Análisis de redes sociales** (Twitter, Reddit, Stocktwits)
- **Análisis de sentimiento con IA** (FinTwitBERT + Claude)
- **Validación técnica** (RSI, MACD, ADX, volumen)
- **Gestión de riesgo** (circuit breakers, position sizing)
- **Ejecución automatizada** via Alpaca API

### Filosofía: Social-First

El sistema detecta oportunidades en redes sociales primero, luego valida técnicamente antes de ejecutar.

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET GATE (Capa 0)                     │
│   Horario OK? │ Volumen OK? │ VIX OK? │ No choppy?          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   COLLECTORS (Capa 1)                       │
│   Twitter │ Reddit │ Stocktwits │ Alpaca News               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ANALYZERS (Capa 2)                        │
│   FinTwitBERT Sentiment │ Ticker Extraction │ Claude AI     │
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
# Alpaca (REQUERIDO)
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_PAPER=true

# Claude AI (REQUERIDO para análisis profundo)
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXX

# Telegram (REQUERIDO para alertas)
TELEGRAM_BOT_TOKEN=1234567890:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TELEGRAM_CHAT_ID=123456789

# Reddit (OPCIONAL)
REDDIT_CLIENT_ID=XXXXXXXXXXXXXX
REDDIT_CLIENT_SECRET=XXXXXXXXXXXXXXXXXXXXXX
REDDIT_USER_AGENT=TradingBot/1.0
```

### 4.2 Archivo de Configuración (config/settings.yaml)

#### Secciones Principales:

**system:** Configuración general
```yaml
system:
  name: "Intraday Trading System"
  mode: "paper"  # paper | live
  timezone: "America/New_York"
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
uv run streamlit run src/dashboard/app.py --server.port 8501
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

### 6.1 Collectors (src/collectors/)

Recolectan mensajes de redes sociales.

```python
from src.collectors import CollectorManager, TwitterCollector

# Crear collector
twitter = TwitterCollector(
    accounts_to_follow=["unusual_whales", "FirstSquawk"],
    refresh_interval=15,
)

# Crear manager
manager = CollectorManager([twitter])

# Agregar callback para mensajes
manager.add_callback(lambda msg: print(msg.content))

# Iniciar
await manager.run()
```

### 6.2 Analyzers (src/analyzers/)

Analizan sentimiento y extraen información.

```python
from src.analyzers import SentimentAnalyzer, ClaudeAnalyzer

# Sentiment con FinTwitBERT
sentiment = SentimentAnalyzer()
result = await sentiment.analyze("NVDA looking bullish! 🚀")
# result.direction = "bullish", result.confidence = 0.89

# Análisis profundo con Claude
claude = ClaudeAnalyzer()
analysis = await claude.analyze_catalyst(message)
# analysis.catalyst_type, analysis.risk_factors, etc.
```

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
from src.scoring import ScoringEngine

engine = ScoringEngine(settings)
recommendation = engine.calculate_recommendation(
    sentiment_result=sentiment,
    technical_result=technical,
    source="unusual_whales"
)

# recommendation.final_score = 85
# recommendation.tier = "strong"
# recommendation.position_size_percent = 100
# recommendation.stop_loss, recommendation.take_profit
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
from src.execution import ExecutionManager

execution = ExecutionManager(alpaca_client, risk_manager)
result = await execution.execute_signal(signal)

# result.order_id, result.filled_price, result.status
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

1. **Live Signals** - Señales en tiempo real
2. **Positions** - Posiciones abiertas
3. **Journal** - Historial de trades
4. **Analytics** - Métricas y análisis

### 7.3 Alertas en Dashboard

- 🟢 **Verde**: Sistema funcionando normal
- 🟡 **Amarillo**: Advertencia (VIX elevado, etc.)
- 🔴 **Rojo**: Circuit breaker activado

---

## 8. Alertas y Notificaciones

### 8.1 Tipos de Alertas (Telegram)

| Tipo | Descripción |
|------|-------------|
| `new_signal` | Nueva señal detectada |
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

### 8.3 Pre-Market Checklist

Cada mañana a las 9:00 AM (configurable):

```
📋 Pre-Market Checklist

☐ Economic calendar reviewed
☐ Overnight news checked
☐ Watchlist prepared
☐ Mental state: focused
☐ Risk parameters confirmed

[Mark All Complete] [Skip Today]
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

### 11.3 Error de Claude API

```
Error: Anthropic API authentication failed
```

**Solución:**
1. Verificar `ANTHROPIC_API_KEY`
2. Verificar que la cuenta tiene créditos
3. El sistema funciona sin Claude (usa solo FinTwitBERT)

### 11.4 Market Gate Siempre Cerrado

```
Gate status: CLOSED - Outside trading hours
```

**Verificar:**
1. Horario del sistema (timezone: America/New_York)
2. Que sea día de mercado (no fin de semana/feriado)
3. Configuración `trading_start` y `trading_end`

### 11.5 Sin Señales Generadas

**Posibles causas:**
1. Collectors no conectados
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

### 11.6 Tests Fallando

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
uv run streamlit run src/dashboard/app.py

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

*Manual generado: 2026-01-17*
*Versión: 1.0.0*
