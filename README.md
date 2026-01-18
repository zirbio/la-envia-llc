# Intraday Trading System

Sistema de trading intradiario impulsado por IA que combina análisis de redes sociales, análisis de sentimiento y validación técnica para ejecutar trades automatizados.

## Características

- **Morning Research Agent**: Daily Briefs pre-mercado con ideas de trading (12:00 y 15:00 Madrid)
- **Señales Sociales via Grok**: Análisis de X/Twitter con sentiment nativo
- **Análisis Profundo con Claude**: Catalizadores, riesgos y contexto de mercado
- **Validación Técnica**: RSI, MACD, ADX, volumen, flujo de opciones
- **Gestión de Riesgo**: Circuit breakers por trade (1%), diario (3%) y semanal (6%)
- **Ejecución Automatizada**: Integración con Alpaca API (paper y live)
- **Dashboard en Tiempo Real**: Monitoreo via Streamlit (incluyendo Research)
- **Notificaciones**: Alertas y Daily Briefs via Telegram

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

## Estructura del Proyecto

```
src/
├── research/        # Morning Research Agent (Daily Briefs, Ideas)
├── collectors/      # Recolección de datos (Grok/xAI)
├── analyzers/       # Análisis con Claude AI
├── validators/      # Validación técnica (RSI, MACD, ADX)
├── scoring/         # Sistema de puntuación (SignalScorer)
├── gate/            # Filtro de condiciones de mercado
├── risk/            # Gestión de riesgo y circuit breakers
├── execution/       # Ejecución via Alpaca (TradeExecutor)
├── orchestrator/    # Coordinador del pipeline
├── journal/         # Registro de trades y métricas
├── notifications/   # Alertas Telegram
├── dashboard/       # Interfaz Streamlit (5 páginas)
├── config/          # Sistema de configuración Pydantic
├── models/          # Modelos de datos
└── validation/      # Escenarios de validación E2E

tests/               # 747 tests (unit, integration, validation)
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

1. **Home** - Vista general del sistema y estado
2. **Monitoreo** - Posiciones, señales, circuit breakers en tiempo real
3. **Análisis** - Métricas de rendimiento, gráficos, patrones
4. **Control** - Parámetros de riesgo, control del gate
5. **Research** - Daily Briefs, ideas de trading, historial

## Documentación

- [Manual de Uso](docs/MANUAL-DE-USO.md) - Guía completa del sistema
- [Siguientes Pasos](docs/SIGUIENTES-PASOS.md) - Configuración inicial y APIs
- [Documentos de Diseño](docs/plans/) - Arquitectura de cada fase

## Licencia

Propietario - La Envia LLC
