"""Alerts page - Alert center with history."""
import streamlit as st

from src.dashboard.models import AlertLevel, AlertType
from src.dashboard.state import DashboardState

st.set_page_config(page_title="Alertas | Trading Dashboard", page_icon="🔔", layout="wide")

st.title("🔔 Centro de Alertas")

state = DashboardState.get_instance()

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
    filter_type = st.selectbox("Filtrar por tipo", options=["Todas", "Trades", "Circuit Breakers", "Gate", "Errores"], index=0)

st.divider()

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

st.subheader("📋 Historial de Alertas")

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
        if alert.level == AlertLevel.ERROR:
            container = st.error
        elif alert.level == AlertLevel.WARNING:
            container = st.warning
        else:
            container = st.info

        time_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        read_marker = "" if alert.read else "🆕 "
        symbol_str = f"[{alert.symbol}] " if alert.symbol else ""
        container(f"{read_marker}**{alert.title}**\n\n{symbol_str}{alert.message}\n\n_{time_str}_")
else:
    st.info("No hay alertas que mostrar", icon="✅")

st.divider()

with st.expander("🧪 Herramientas de Desarrollo"):
    if st.button("Agregar alerta de prueba (Trade)"):
        state.add_alert(AlertType.TRADE_EXECUTED, AlertLevel.INFO, "Trade Ejecutado", "Compra de 100 NVDA @ $140.00", "NVDA")
        st.rerun()

    if st.button("Agregar alerta de prueba (Error)"):
        state.add_alert(AlertType.SYSTEM_ERROR, AlertLevel.ERROR, "Error de Conexión", "No se pudo conectar con Alpaca API")
        st.rerun()
