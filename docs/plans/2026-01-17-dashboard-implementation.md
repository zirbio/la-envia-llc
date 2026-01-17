# Streamlit Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-page Streamlit dashboard for real-time monitoring and performance analysis of the trading system.

**Architecture:** Multi-page Streamlit app with a DashboardState singleton connecting to existing system managers. Pages for monitoring, analysis, control, and alerts. Components are reusable UI elements.

**Tech Stack:** Streamlit 1.53+, Plotly 6.5+, Pandas, existing system modules (JournalManager, RiskManager, etc.)

---

## Task 1: Create DashboardSettings

**Files:**
- Create: `src/dashboard/__init__.py`
- Create: `src/dashboard/settings.py`
- Create: `tests/dashboard/__init__.py`
- Create: `tests/dashboard/test_settings.py`

**Step 1: Create module init files**

```python
# src/dashboard/__init__.py
"""Streamlit dashboard for trading system monitoring and analysis."""

# tests/dashboard/__init__.py
# (empty file)
```

**Step 2: Write the failing test**

```python
# tests/dashboard/test_settings.py
"""Tests for DashboardSettings."""
import pytest
from src.dashboard.settings import DashboardSettings


class TestDashboardSettings:
    """Tests for DashboardSettings."""

    def test_default_settings(self) -> None:
        """Default settings should have sensible values."""
        settings = DashboardSettings()

        assert settings.refresh_interval_seconds == 5
        assert settings.max_signals_displayed == 10
        assert settings.max_alerts_displayed == 50
        assert settings.theme == "dark"

    def test_custom_settings(self) -> None:
        """Custom settings should override defaults."""
        settings = DashboardSettings(
            refresh_interval_seconds=10,
            max_signals_displayed=20,
            theme="light",
        )

        assert settings.refresh_interval_seconds == 10
        assert settings.max_signals_displayed == 20
        assert settings.theme == "light"

    def test_refresh_interval_validation(self) -> None:
        """Refresh interval must be positive."""
        with pytest.raises(ValueError):
            DashboardSettings(refresh_interval_seconds=0)

    def test_theme_validation(self) -> None:
        """Theme must be light or dark."""
        with pytest.raises(ValueError):
            DashboardSettings(theme="invalid")
```

**Step 3: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/test_settings.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/dashboard/settings.py
"""Settings for the Streamlit dashboard."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DashboardSettings(BaseModel):
    """Configuration for the dashboard."""

    refresh_interval_seconds: int = Field(default=5, gt=0)
    max_signals_displayed: int = Field(default=10, gt=0)
    max_alerts_displayed: int = Field(default=50, gt=0)
    theme: Literal["light", "dark"] = "dark"

    @field_validator("refresh_interval_seconds")
    @classmethod
    def validate_refresh_interval(cls, v: int) -> int:
        """Validate refresh interval is positive."""
        if v <= 0:
            raise ValueError("refresh_interval_seconds must be positive")
        return v
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/test_settings.py -v`
Expected: PASS (4 tests)

**Step 6: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/ tests/dashboard/ && git commit -m "feat(dashboard): add DashboardSettings with validation"
```

---

## Task 2: Create AlertEvent Model

**Files:**
- Create: `src/dashboard/models.py`
- Create: `tests/dashboard/test_models.py`

**Step 1: Write the failing test**

```python
# tests/dashboard/test_models.py
"""Tests for dashboard models."""
from datetime import datetime

from src.dashboard.models import AlertEvent, AlertLevel, AlertType


class TestAlertEvent:
    """Tests for AlertEvent model."""

    def test_create_alert_event(self) -> None:
        """Should create an alert event with all fields."""
        event = AlertEvent(
            timestamp=datetime(2026, 1, 17, 9, 30, 0),
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Trade Executed",
            message="Bought 100 NVDA @ $140.00",
            symbol="NVDA",
        )

        assert event.alert_type == AlertType.TRADE_EXECUTED
        assert event.level == AlertLevel.INFO
        assert event.symbol == "NVDA"
        assert not event.read

    def test_alert_types(self) -> None:
        """Should have all expected alert types."""
        assert AlertType.TRADE_EXECUTED.value == "trade_executed"
        assert AlertType.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert AlertType.GATE_CHANGE.value == "gate_change"
        assert AlertType.SYSTEM_ERROR.value == "system_error"

    def test_alert_levels(self) -> None:
        """Should have all expected alert levels."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/dashboard/models.py
"""Data models for the dashboard."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertType(Enum):
    """Types of alerts."""

    TRADE_EXECUTED = "trade_executed"
    CIRCUIT_BREAKER = "circuit_breaker"
    GATE_CHANGE = "gate_change"
    SYSTEM_ERROR = "system_error"


class AlertLevel(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class AlertEvent:
    """An alert event for the dashboard."""

    timestamp: datetime
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    symbol: str | None = None
    read: bool = field(default=False)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/test_models.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/models.py tests/dashboard/test_models.py && git commit -m "feat(dashboard): add AlertEvent model with types and levels"
```

---

## Task 3: Create DashboardState

**Files:**
- Create: `src/dashboard/state.py`
- Create: `tests/dashboard/test_state.py`

**Step 1: Write the failing test**

```python
# tests/dashboard/test_state.py
"""Tests for DashboardState."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.dashboard.models import AlertEvent, AlertLevel, AlertType
from src.dashboard.state import DashboardState


class TestDashboardState:
    """Tests for DashboardState."""

    def test_singleton_pattern(self) -> None:
        """DashboardState should return same instance."""
        state1 = DashboardState.get_instance()
        state2 = DashboardState.get_instance()

        assert state1 is state2

    def test_add_alert(self) -> None:
        """Should add alerts to history."""
        state = DashboardState.get_instance()
        state.clear_alerts()

        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Test message",
        )

        assert len(state.alerts) == 1
        assert state.alerts[0].title == "Test Alert"

    def test_alerts_limited_by_max(self) -> None:
        """Should limit alerts to max_alerts setting."""
        state = DashboardState.get_instance()
        state.clear_alerts()
        state._max_alerts = 3

        for i in range(5):
            state.add_alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.ERROR,
                title=f"Alert {i}",
                message=f"Message {i}",
            )

        assert len(state.alerts) == 3
        # Most recent alerts kept
        assert state.alerts[0].title == "Alert 4"

    def test_clear_alerts(self) -> None:
        """Should clear all alerts."""
        state = DashboardState.get_instance()
        state.add_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="Test",
            message="Test",
        )
        state.clear_alerts()

        assert len(state.alerts) == 0

    def test_unread_count(self) -> None:
        """Should count unread alerts."""
        state = DashboardState.get_instance()
        state.clear_alerts()

        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Alert 1",
            message="Message 1",
        )
        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Alert 2",
            message="Message 2",
        )

        assert state.unread_count == 2

        state.alerts[0].read = True
        assert state.unread_count == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/test_state.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/dashboard/state.py
"""Dashboard state management."""
from datetime import datetime
from typing import ClassVar

from src.dashboard.models import AlertEvent, AlertLevel, AlertType


class DashboardState:
    """Singleton state manager for the dashboard.

    Maintains alert history and provides connection points
    for system managers.
    """

    _instance: ClassVar["DashboardState | None"] = None

    def __init__(self) -> None:
        """Initialize dashboard state."""
        self._alerts: list[AlertEvent] = []
        self._max_alerts: int = 50

    @classmethod
    def get_instance(cls) -> "DashboardState":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def alerts(self) -> list[AlertEvent]:
        """Get all alerts, most recent first."""
        return self._alerts

    @property
    def unread_count(self) -> int:
        """Count of unread alerts."""
        return sum(1 for alert in self._alerts if not alert.read)

    def add_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        symbol: str | None = None,
    ) -> None:
        """Add a new alert to history."""
        alert = AlertEvent(
            timestamp=datetime.now(),
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            symbol=symbol,
        )
        self._alerts.insert(0, alert)

        # Trim to max alerts
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[: self._max_alerts]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts = []

    def mark_all_read(self) -> None:
        """Mark all alerts as read."""
        for alert in self._alerts:
            alert.read = True
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/test_state.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/state.py tests/dashboard/test_state.py && git commit -m "feat(dashboard): add DashboardState singleton for alert management"
```

---

## Task 4: Create Home Page

**Files:**
- Create: `src/dashboard/Home.py`
- Create: `src/dashboard/pages/__init__.py`

**Note:** Streamlit pages are not typically unit tested. We'll create the page and test manually.

**Step 1: Create pages directory**

```python
# src/dashboard/pages/__init__.py
"""Dashboard pages."""
```

**Step 2: Create Home page**

```python
# src/dashboard/Home.py
"""Home page - Dashboard landing with summary."""
import streamlit as st

from src.dashboard.state import DashboardState

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading Dashboard")

# Get state
state = DashboardState.get_instance()

# Summary metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Sistema",
        value="STOPPED",
        delta=None,
    )

with col2:
    st.metric(
        label="Posiciones Abiertas",
        value="0",
        delta=None,
    )

with col3:
    st.metric(
        label="P&L Hoy",
        value="$0.00",
        delta="0%",
    )

with col4:
    unread = state.unread_count
    st.metric(
        label="Alertas",
        value=str(unread),
        delta="sin leer" if unread > 0 else None,
    )

st.divider()

# Navigation cards
st.subheader("Navegación Rápida")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_Monitoreo.py", label="🎯 Monitoreo", icon="🎯")
    st.caption("Posiciones y señales en tiempo real")

with col2:
    st.page_link("pages/2_Analisis.py", label="📊 Análisis", icon="📊")
    st.caption("Métricas y rendimiento")

with col3:
    st.page_link("pages/3_Control.py", label="⚙️ Control", icon="⚙️")
    st.caption("Controles del sistema")

with col4:
    st.page_link("pages/4_Alertas.py", label="🔔 Alertas", icon="🔔")
    st.caption("Centro de notificaciones")

st.divider()

# Recent alerts
st.subheader("Alertas Recientes")

if state.alerts:
    for alert in state.alerts[:3]:
        icon = "ℹ️" if alert.level.value == "info" else "⚠️" if alert.level.value == "warning" else "❌"
        st.info(f"{icon} **{alert.title}** - {alert.message}", icon=icon)
else:
    st.info("No hay alertas recientes", icon="✅")
```

**Step 3: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/Home.py src/dashboard/pages/__init__.py && git commit -m "feat(dashboard): add Home page with summary and navigation"
```

---

## Task 5: Create Monitoring Page

**Files:**
- Create: `src/dashboard/pages/1_Monitoreo.py`

**Step 1: Create monitoring page**

```python
# src/dashboard/pages/1_Monitoreo.py
"""Monitoring page - Real-time positions and signals."""
import streamlit as st

from src.dashboard.state import DashboardState

st.set_page_config(
    page_title="Monitoreo | Trading Dashboard",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Monitoreo en Tiempo Real")

# Get state
state = DashboardState.get_instance()

# Status bar
col1, col2, col3 = st.columns(3)

with col1:
    status_color = "🟢" if False else "🔴"  # TODO: Connect to orchestrator
    st.markdown(f"### {status_color} Sistema: STOPPED")

with col2:
    gate_color = "🟢" if True else "🔴"  # TODO: Connect to market gate
    st.markdown(f"### {gate_color} Gate: OPEN")

with col3:
    from datetime import datetime
    st.markdown(f"### ⏱ {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# Positions section
st.subheader("📋 Posiciones Abiertas")

# Placeholder for positions - will be connected to TradeExecutor
positions_data = []  # TODO: Get from executor.get_positions()

if positions_data:
    import pandas as pd
    df = pd.DataFrame(positions_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # Total P&L
    total_pnl = sum(p.get("pnl", 0) for p in positions_data)
    pnl_color = "green" if total_pnl >= 0 else "red"
    st.markdown(f"**P&L Total:** :{pnl_color}[${total_pnl:,.2f}]")
else:
    st.info("No hay posiciones abiertas", icon="📭")

st.divider()

# Signals section
st.subheader("📡 Señales Recientes")

# Placeholder for signals - will be connected to orchestrator
signals_data = []  # TODO: Get from orchestrator recent signals

if signals_data:
    for signal in signals_data[:10]:
        score = signal.get("score", 0)
        color = "🟢" if score >= 75 else "🟡" if score >= 60 else "🔴"
        st.markdown(
            f"{signal.get('time', '')} {signal.get('symbol', '')} "
            f"{color} Score: {score} - {signal.get('reason', '')}"
        )
else:
    st.info("No hay señales recientes", icon="📡")

st.divider()

# Circuit breakers section
st.subheader("🛡️ Circuit Breakers")

col1, col2, col3 = st.columns(3)

with col1:
    # Daily loss progress
    daily_loss_pct = 0.4  # TODO: Get from risk manager
    daily_loss_limit = 1000  # TODO: Get from settings
    st.progress(daily_loss_pct, text=f"Daily Loss: {daily_loss_pct*100:.0f}%")

with col2:
    # Consecutive losses
    consecutive = 1  # TODO: Get from risk manager
    max_consecutive = 3
    st.metric("Pérdidas Consecutivas", f"{consecutive}/{max_consecutive}")

with col3:
    # Drawdown
    drawdown = 2.5  # TODO: Get from risk manager
    max_drawdown = 5.0
    st.metric("Drawdown", f"{drawdown:.1f}%/{max_drawdown:.1f}%")

# Auto-refresh
if st.checkbox("Auto-refresh (5s)", value=False):
    import time
    time.sleep(5)
    st.rerun()
```

**Step 2: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/pages/1_Monitoreo.py && git commit -m "feat(dashboard): add Monitoring page with positions and signals"
```

---

## Task 6: Create Analysis Page

**Files:**
- Create: `src/dashboard/pages/2_Analisis.py`

**Step 1: Create analysis page**

```python
# src/dashboard/pages/2_Analisis.py
"""Analysis page - Performance metrics and charts."""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Análisis | Trading Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Análisis de Rendimiento")

# Period selector
col1, col2 = st.columns([1, 3])

with col1:
    period = st.selectbox(
        "Período",
        options=["Hoy", "Semana", "Mes", "Custom"],
        index=1,
    )

with col2:
    if period == "Custom":
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Desde", value=date.today() - timedelta(days=30))
        with col_end:
            end_date = st.date_input("Hasta", value=date.today())
    else:
        end_date = date.today()
        if period == "Hoy":
            start_date = end_date
        elif period == "Semana":
            start_date = end_date - timedelta(days=7)
        else:  # Mes
            start_date = end_date - timedelta(days=30)

st.divider()

# Metrics row
st.subheader("📈 Métricas Principales")

col1, col2, col3, col4 = st.columns(4)

# TODO: Get real metrics from JournalManager
with col1:
    st.metric("Win Rate", "65%", delta="+5%")

with col2:
    st.metric("Profit Factor", "2.1", delta="+0.3")

with col3:
    st.metric("Expectancy", "0.85R", delta="+0.1R")

with col4:
    st.metric("Total P&L", "$1,250", delta="+$350")

st.divider()

# Charts
st.subheader("📉 Gráficos")

tab1, tab2, tab3 = st.tabs(["Equity Curve", "P&L por Día", "Win Rate por Hora"])

with tab1:
    # Placeholder equity curve
    # TODO: Get real data from journal
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    equity = [10000 + i * 50 + (i % 3 - 1) * 100 for i in range(len(dates))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        mode="lines",
        name="Equity",
        line=dict(color="green", width=2),
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Fecha",
        yaxis_title="Equity ($)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Placeholder P&L by day
    # TODO: Get real data from journal
    pnl_data = {
        "Día": ["Lun", "Mar", "Mié", "Jue", "Vie"],
        "P&L": [150, -50, 200, 100, -30],
    }
    df = pd.DataFrame(pnl_data)

    colors = ["green" if x >= 0 else "red" for x in df["P&L"]]
    fig = go.Figure(data=[
        go.Bar(x=df["Día"], y=df["P&L"], marker_color=colors)
    ])
    fig.update_layout(
        title="P&L por Día",
        xaxis_title="Día",
        yaxis_title="P&L ($)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Placeholder win rate by hour
    # TODO: Get real data from pattern analyzer
    hours = list(range(9, 16))
    win_rates = [0.6, 0.7, 0.65, 0.55, 0.7, 0.75, 0.5]

    fig = go.Figure(data=[
        go.Bar(x=hours, y=win_rates, marker_color="blue")
    ])
    fig.update_layout(
        title="Win Rate por Hora",
        xaxis_title="Hora",
        yaxis_title="Win Rate",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Patterns section
st.subheader("🔍 Patrones Identificados")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Mejores:**")
    st.success("⏰ Mejor hora: 10:00 AM (75% WR)")
    st.success("📊 Mejor símbolo: NVDA (+$450)")
    st.success("📅 Mejor día: Jueves (80% WR)")

with col2:
    st.markdown("**A Mejorar:**")
    st.warning("⏰ Peor hora: 3:00 PM (45% WR)")
    st.warning("📊 Peor símbolo: TSLA (-$120)")
    st.warning("📅 Peor día: Viernes (40% WR)")
```

**Step 2: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/pages/2_Analisis.py && git commit -m "feat(dashboard): add Analysis page with metrics and charts"
```

---

## Task 7: Create Control Page

**Files:**
- Create: `src/dashboard/pages/3_Control.py`

**Step 1: Create control page**

```python
# src/dashboard/pages/3_Control.py
"""Control page - System controls and settings."""
import streamlit as st

from src.dashboard.state import DashboardState

st.set_page_config(
    page_title="Control | Trading Dashboard",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Control del Sistema")

state = DashboardState.get_instance()

# System controls
st.subheader("🎛️ Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    is_running = False  # TODO: Get from orchestrator.is_running
    if is_running:
        if st.button("🛑 Detener Sistema", type="primary", use_container_width=True):
            # TODO: Call orchestrator.stop()
            st.warning("Sistema detenido")
            state.add_alert(
                alert_type=state.AlertType.SYSTEM_ERROR,
                level=state.AlertLevel.WARNING,
                title="Sistema Detenido",
                message="El sistema fue detenido manualmente",
            )
    else:
        if st.button("▶️ Iniciar Sistema", type="primary", use_container_width=True):
            # TODO: Call orchestrator.start()
            st.success("Sistema iniciado")

with col2:
    st.metric("Estado", "STOPPED" if not is_running else "RUNNING")

with col3:
    st.metric("Uptime", "0:00:00")  # TODO: Track uptime

st.divider()

# Position controls
st.subheader("📋 Posiciones")

positions = []  # TODO: Get from executor.get_positions()

if positions:
    for pos in positions:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{pos['symbol']}** - {pos['quantity']} shares @ ${pos['entry_price']:.2f}")
        with col2:
            pnl = pos.get('pnl', 0)
            color = "green" if pnl >= 0 else "red"
            st.markdown(f"P&L: :{color}[${pnl:,.2f}]")
        with col3:
            if st.button(f"Cerrar {pos['symbol']}", key=f"close_{pos['symbol']}"):
                # TODO: Call executor.close_position(pos['symbol'])
                st.warning(f"Posición {pos['symbol']} cerrada")

    st.divider()
    if st.button("🚨 Cerrar TODAS las Posiciones", type="secondary"):
        # TODO: Call executor.close_all_positions()
        st.error("Todas las posiciones cerradas")
else:
    st.info("No hay posiciones abiertas", icon="📭")

st.divider()

# Risk parameters
st.subheader("⚠️ Parámetros de Riesgo")

col1, col2 = st.columns(2)

with col1:
    max_position = st.slider(
        "Max Position Size (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Tamaño máximo de posición como % del portfolio",
    )

    max_daily_loss = st.slider(
        "Max Daily Loss (%)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Pérdida máxima diaria permitida",
    )

with col2:
    max_consecutive = st.number_input(
        "Max Consecutive Losses",
        min_value=1,
        max_value=10,
        value=3,
        help="Número de pérdidas consecutivas antes de pausar",
    )

    max_drawdown = st.slider(
        "Max Drawdown (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Drawdown máximo permitido",
    )

if st.button("💾 Guardar Parámetros"):
    # TODO: Update risk manager settings
    st.success("Parámetros actualizados")

st.divider()

# Circuit breakers
st.subheader("🛡️ Circuit Breakers")

col1, col2, col3 = st.columns(3)

with col1:
    daily_loss_cb = st.toggle("Daily Loss Breaker", value=True)
    st.caption("Pausa si pérdida diaria excede límite")

with col2:
    consecutive_cb = st.toggle("Consecutive Loss Breaker", value=True)
    st.caption("Pausa después de N pérdidas seguidas")

with col3:
    drawdown_cb = st.toggle("Drawdown Breaker", value=True)
    st.caption("Pausa si drawdown excede límite")

st.divider()

# Market gate
st.subheader("🚪 Market Gate")

col1, col2 = st.columns(2)

with col1:
    gate_status = "OPEN"  # TODO: Get from market_gate.status
    st.metric("Estado Actual", gate_status)

with col2:
    override = st.selectbox(
        "Override Manual",
        options=["Auto", "Force OPEN", "Force CLOSED"],
        index=0,
    )
    if override != "Auto":
        st.warning(f"Gate forzado a: {override}")
```

**Step 2: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/pages/3_Control.py && git commit -m "feat(dashboard): add Control page with system and risk controls"
```

---

## Task 8: Create Alerts Page

**Files:**
- Create: `src/dashboard/pages/4_Alertas.py`

**Step 1: Create alerts page**

```python
# src/dashboard/pages/4_Alertas.py
"""Alerts page - Alert center with history."""
import streamlit as st

from src.dashboard.models import AlertLevel, AlertType
from src.dashboard.state import DashboardState

st.set_page_config(
    page_title="Alertas | Trading Dashboard",
    page_icon="🔔",
    layout="wide",
)

st.title("🔔 Centro de Alertas")

state = DashboardState.get_instance()

# Controls
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("✅ Marcar todas como leídas"):
        state.mark_all_read()
        st.rerun()

with col2:
    if st.button("🗑️ Limpiar alertas"):
        state.clear_alerts()
        st.rerun()

with col3:
    filter_type = st.selectbox(
        "Filtrar por tipo",
        options=["Todas", "Trades", "Circuit Breakers", "Gate", "Errores"],
        index=0,
    )

st.divider()

# Stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total", len(state.alerts))

with col2:
    st.metric("Sin Leer", state.unread_count)

with col3:
    errors = sum(1 for a in state.alerts if a.level == AlertLevel.ERROR)
    st.metric("Errores", errors)

with col4:
    warnings = sum(1 for a in state.alerts if a.level == AlertLevel.WARNING)
    st.metric("Advertencias", warnings)

st.divider()

# Alert list
st.subheader("📋 Historial de Alertas")

# Filter alerts
alerts = state.alerts
if filter_type == "Trades":
    alerts = [a for a in alerts if a.alert_type == AlertType.TRADE_EXECUTED]
elif filter_type == "Circuit Breakers":
    alerts = [a for a in alerts if a.alert_type == AlertType.CIRCUIT_BREAKER]
elif filter_type == "Gate":
    alerts = [a for a in alerts if a.alert_type == AlertType.GATE_CHANGE]
elif filter_type == "Errores":
    alerts = [a for a in alerts if a.alert_type == AlertType.SYSTEM_ERROR]

if alerts:
    for alert in alerts:
        # Determine icon and color
        if alert.level == AlertLevel.ERROR:
            icon = "🔴"
            container = st.error
        elif alert.level == AlertLevel.WARNING:
            icon = "🟡"
            container = st.warning
        else:
            icon = "🔵"
            container = st.info

        # Format timestamp
        time_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Display alert
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                read_marker = "" if alert.read else "🆕 "
                symbol_str = f"[{alert.symbol}] " if alert.symbol else ""
                container(
                    f"{read_marker}{icon} **{alert.title}**\n\n"
                    f"{symbol_str}{alert.message}\n\n"
                    f"_{time_str}_"
                )
            with col2:
                if not alert.read:
                    if st.button("Marcar leída", key=f"read_{id(alert)}"):
                        alert.read = True
                        st.rerun()
else:
    st.info("No hay alertas que mostrar", icon="✅")

# Demo: Add test alerts button (for development)
st.divider()
with st.expander("🧪 Herramientas de Desarrollo"):
    if st.button("Agregar alerta de prueba (Trade)"):
        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Trade Ejecutado",
            message="Compra de 100 NVDA @ $140.00",
            symbol="NVDA",
        )
        st.rerun()

    if st.button("Agregar alerta de prueba (Error)"):
        state.add_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="Error de Conexión",
            message="No se pudo conectar con Alpaca API",
        )
        st.rerun()

    if st.button("Agregar alerta de prueba (Circuit Breaker)"):
        state.add_alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            level=AlertLevel.WARNING,
            title="Circuit Breaker Activado",
            message="Daily loss limit alcanzado (3%)",
        )
        st.rerun()
```

**Step 2: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/pages/4_Alertas.py && git commit -m "feat(dashboard): add Alerts page with history and filters"
```

---

## Task 9: Update Module Exports

**Files:**
- Modify: `src/dashboard/__init__.py`

**Step 1: Update exports**

```python
# src/dashboard/__init__.py
"""Streamlit dashboard for trading system monitoring and analysis."""

from src.dashboard.models import AlertEvent, AlertLevel, AlertType
from src.dashboard.settings import DashboardSettings
from src.dashboard.state import DashboardState

__all__ = [
    "AlertEvent",
    "AlertLevel",
    "AlertType",
    "DashboardSettings",
    "DashboardState",
]
```

**Step 2: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && git add src/dashboard/__init__.py && git commit -m "feat(dashboard): update module exports"
```

---

## Task 10: Final Test Suite Run

**Step 1: Run all dashboard tests**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/dashboard/ -v`
Expected: All tests pass

**Step 2: Run full test suite**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard && uv run pytest tests/ -q`
Expected: All tests pass (600+ tests)

---

## Running the Dashboard

After implementation, run the dashboard with:

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase11-dashboard
uv run streamlit run src/dashboard/Home.py
```

The dashboard will be available at `http://localhost:8501`
