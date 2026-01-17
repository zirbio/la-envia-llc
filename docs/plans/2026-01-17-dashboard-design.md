# Streamlit Dashboard Design (Phase 11)

## Overview

Dashboard multi-página para monitoreo en tiempo real y análisis de rendimiento del sistema de trading.

## Decisiones de Diseño

- **Navegación**: Multi-página nativo de Streamlit
- **Monitoreo**: Posiciones abiertas como prioridad, luego señales y circuit breakers
- **Análisis**: Selector flexible de período (Hoy/Semana/Mes/Custom)
- **Control**: Completo - Start/Stop, cerrar posiciones, ajustar risk params, circuit breakers
- **Alertas**: Centro de alertas con historial de eventos

## Estructura de Archivos

```
src/dashboard/
├── __init__.py
├── Home.py                 # Landing page con resumen
├── pages/
│   ├── 1_Monitoreo.py     # Real-time monitoring
│   ├── 2_Analisis.py      # Performance analysis
│   ├── 3_Control.py       # System control
│   └── 4_Alertas.py       # Alert center
├── components/
│   ├── __init__.py
│   ├── position_card.py   # Position display component
│   ├── signal_feed.py     # Signal feed component
│   ├── metrics_charts.py  # Plotly charts for metrics
│   └── alert_banner.py    # Alert banner component
├── state.py               # DashboardState singleton
└── settings.py            # DashboardSettings
```

## Páginas

### 1. Home (Landing)

- Resumen rápido: P&L del día, posiciones abiertas, estado del sistema
- Links a las páginas principales
- Últimas 3 alertas

### 2. Monitoreo (Real-time)

```
┌─────────────────────────────────────────────────────┐
│ 🟢 Sistema: RUNNING    🟡 Gate: OPEN    ⏱ 09:45:32 │
├─────────────────────────────────────────────────────┤
│ POSICIONES ABIERTAS                        P&L: +$850│
│ ┌─────────┬───────┬────────┬────────┬─────────────┐ │
│ │ Symbol  │ Side  │ Entry  │ Current│ P&L   │ R   │ │
│ │ NVDA    │ LONG  │ $140.00│ $142.50│ +$250 │+1.2R│ │
│ └─────────┴───────┴────────┴────────┴─────────────┘ │
├─────────────────────────────────────────────────────┤
│ SEÑALES RECIENTES (últimas 10)                      │
│ 09:44 TSLA 🟢 Score: 82 - Strong bullish sentiment  │
├─────────────────────────────────────────────────────┤
│ CIRCUIT BREAKERS                                    │
│ Daily Loss: ████░░░░░░ 40% ($400/$1000)            │
│ Consecutive: 1/3 ✓    Drawdown: 2.5%/5% ✓          │
└─────────────────────────────────────────────────────┘
```

- Auto-refresh cada 5 segundos (configurable)
- Colores condicionales para P&L (verde/rojo)
- Progress bars para circuit breakers

### 3. Análisis (Performance)

- Selector de período: Hoy / Semana / Mes / Custom
- Métricas principales: Win Rate, Profit Factor, Expectancy, Sharpe
- Gráficos Plotly:
  - Equity curve
  - P&L por día (bar chart)
  - Win rate por hora (heatmap)
  - Distribución de R-multiples
- Tabla de trades con filtros
- Patrones identificados (best/worst hours, symbols, setups)

### 4. Control

- **Sistema**: Botones Start/Stop orchestrator
- **Posiciones**: Botón para cerrar posición individual o todas
- **Risk Parameters**: Sliders para ajustar en vivo:
  - max_position_size_percent
  - max_daily_loss_percent
  - max_consecutive_losses
- **Circuit Breakers**: Toggle para activar/desactivar cada uno
- **Market Gate**: Override manual (force open/close)

### 5. Alertas

- Lista cronológica de eventos:
  - Trade ejecutado (entry/exit)
  - Circuit breaker activado/desactivado
  - Market gate cambio de estado
  - Errores del sistema
- Filtros por tipo y fecha
- Botón para limpiar/marcar como leídas

## Integración

```python
# DashboardState conecta con los managers existentes
class DashboardState:
    def __init__(self):
        self.orchestrator: TradingOrchestrator
        self.journal: JournalManager
        self.risk_manager: RiskManager
        self.market_gate: MarketGate
        self.executor: TradeExecutor
        self.alert_history: list[AlertEvent]
```

## Dependencias

- streamlit >= 1.30
- plotly >= 5.18
- pandas >= 2.0

## Settings

```python
class DashboardSettings(BaseModel):
    refresh_interval_seconds: int = 5
    max_signals_displayed: int = 10
    max_alerts_displayed: int = 50
    theme: Literal["light", "dark"] = "dark"
```
