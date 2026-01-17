"""Home page - Dashboard landing with summary."""
import streamlit as st

from src.dashboard.state import DashboardState

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading Dashboard")

state = DashboardState.get_instance()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Sistema", value="STOPPED")

with col2:
    st.metric(label="Posiciones Abiertas", value="0")

with col3:
    st.metric(label="P&L Hoy", value="$0.00", delta="0%")

with col4:
    unread = state.unread_count
    st.metric(label="Alertas", value=str(unread), delta="sin leer" if unread > 0 else None)

st.divider()

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

st.subheader("Alertas Recientes")

if state.alerts:
    for alert in state.alerts[:3]:
        icon = "ℹ️" if alert.level.value == "info" else "⚠️" if alert.level.value == "warning" else "❌"
        st.info(f"{icon} **{alert.title}** - {alert.message}")
else:
    st.info("No hay alertas recientes", icon="✅")
