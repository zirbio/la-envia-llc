# Intraday Trading System

Sistema de trading intradiario impulsado por IA que combina análisis de redes sociales, análisis de sentimiento y validación técnica para ejecutar trades automatizados.

## Características

- **Análisis Social-First**: Detecta oportunidades en Twitter, Reddit y Stocktwits
- **Análisis de Sentimiento**: FinTwitBERT para análisis rápido + Claude AI para análisis profundo
- **Validación Técnica**: RSI, MACD, ADX, volumen, flujo de opciones
- **Gestión de Riesgo**: Circuit breakers por trade (1%), diario (3%) y semanal (6%)
- **Ejecución Automatizada**: Integración con Alpaca API (paper y live)
- **Dashboard en Tiempo Real**: Monitoreo via Streamlit
- **Notificaciones**: Alertas via Telegram

## Requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes)

## Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd intraday-trading-system

# Instalar dependencias
uv sync

# Copiar configuración de ejemplo
cp .env.example .env

# Editar credenciales
nano .env
```

## Configuración

### Variables de Entorno (.env)

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

### Parámetros del Sistema (config/settings.yaml)

Todos los parámetros configurables están en `config/settings.yaml`:
- Umbrales de scoring (fuerte ≥80, moderado ≥60)
- Límites de riesgo (circuit breakers)
- Períodos de indicadores técnicos
- Horarios de trading y condiciones de mercado

## Uso

```bash
# Iniciar sistema principal
uv run python main.py

# Iniciar dashboard
uv run streamlit run src/dashboard/Home.py --server.port 8501

# Ejecutar tests
uv run pytest

# Ejecutar test específico
uv run pytest tests/path/test_file.py -v
```

## Arquitectura

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

## Estructura del Proyecto

```
src/
├── collectors/      # Recolección de datos sociales
├── analyzers/       # Análisis de sentimiento (FinTwitBERT, Claude)
├── validators/      # Validación técnica (RSI, MACD, ADX)
├── scoring/         # Sistema de puntuación (SignalScorer)
├── gate/            # Filtro de condiciones de mercado
├── risk/            # Gestión de riesgo y circuit breakers
├── execution/       # Ejecución via Alpaca (TradeExecutor)
├── orchestrator/    # Coordinador del pipeline
├── journal/         # Registro de trades y métricas
├── notifications/   # Alertas Telegram
├── dashboard/       # Interfaz Streamlit
├── config/          # Sistema de configuración Pydantic
├── models/          # Modelos de datos
└── validation/      # Escenarios de validación E2E

tests/               # 664 tests (unit, integration, validation)
config/              # Archivos YAML de configuración
docs/                # Documentación
```

## Testing

```bash
# Todos los tests
uv run pytest

# Con cobertura
uv run pytest --cov=src --cov-report=html

# Tests de integración
uv run pytest tests/integration/

# Tests de validación E2E
uv run pytest tests/validation/
```

## Dashboard

Acceder a `http://localhost:8501` después de iniciar el dashboard:

1. **Monitoreo** - Posiciones, señales, circuit breakers en tiempo real
2. **Análisis** - Métricas de rendimiento, gráficos, patrones
3. **Control** - Parámetros de riesgo, control del gate
4. **Alertas** - Centro de alertas con historial y filtros

## Documentación

- [Manual de Uso](docs/MANUAL-DE-USO.md) - Guía completa del sistema
- [Siguientes Pasos](docs/SIGUIENTES-PASOS.md) - Configuración inicial y APIs
- [Documentos de Diseño](docs/plans/) - Arquitectura de cada fase

## Licencia

Propietario - La Envia LLC
