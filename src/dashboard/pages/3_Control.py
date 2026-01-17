"""Control page - System controls and settings."""
import streamlit as st

from src.dashboard.models import AlertLevel, AlertType
from src.dashboard.state import DashboardState

st.set_page_config(page_title="Control | Trading Dashboard", page_icon="⚙️", layout="wide")

st.title("⚙️ Control del Sistema")

state = DashboardState.get_instance()

st.subheader("🎛️ Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Iniciar Sistema", type="primary", use_container_width=True):
        st.success("Sistema iniciado")
        state.add_alert(AlertType.GATE_CHANGE, AlertLevel.INFO, "Sistema Iniciado", "El sistema fue iniciado manualmente")

with col2:
    st.metric("Estado", "STOPPED")

with col3:
    st.metric("Uptime", "0:00:00")

st.divider()

st.subheader("📋 Posiciones")
st.info("No hay posiciones abiertas", icon="📭")

st.divider()

st.subheader("⚠️ Parámetros de Riesgo")

col1, col2 = st.columns(2)

with col1:
    max_position = st.slider("Max Position Size (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    max_daily_loss = st.slider("Max Daily Loss (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

with col2:
    max_consecutive = st.number_input("Max Consecutive Losses", min_value=1, max_value=10, value=3)
    max_drawdown = st.slider("Max Drawdown (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

if st.button("💾 Guardar Parámetros"):
    st.success("Parámetros actualizados")

st.divider()

st.subheader("🛡️ Circuit Breakers")

col1, col2, col3 = st.columns(3)

with col1:
    st.toggle("Daily Loss Breaker", value=True)

with col2:
    st.toggle("Consecutive Loss Breaker", value=True)

with col3:
    st.toggle("Drawdown Breaker", value=True)

st.divider()

st.subheader("🚪 Market Gate")

col1, col2 = st.columns(2)

with col1:
    st.metric("Estado Actual", "OPEN")

with col2:
    st.selectbox("Override Manual", options=["Auto", "Force OPEN", "Force CLOSED"], index=0)
