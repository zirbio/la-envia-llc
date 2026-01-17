"""Monitoring page - Real-time positions and signals."""
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Monitoreo | Trading Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Monitoreo en Tiempo Real")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔴 Sistema: STOPPED")

with col2:
    st.markdown("### 🟢 Gate: OPEN")

with col3:
    st.markdown(f"### ⏱ {datetime.now().strftime('%H:%M:%S')}")

st.divider()

st.subheader("📋 Posiciones Abiertas")
st.info("No hay posiciones abiertas", icon="📭")

st.divider()

st.subheader("📡 Señales Recientes")
st.info("No hay señales recientes", icon="📡")

st.divider()

st.subheader("🛡️ Circuit Breakers")

col1, col2, col3 = st.columns(3)

with col1:
    st.progress(0.4, text="Daily Loss: 40%")

with col2:
    st.metric("Pérdidas Consecutivas", "1/3")

with col3:
    st.metric("Drawdown", "2.5%/5%")

if st.checkbox("Auto-refresh (5s)", value=False):
    import time
    time.sleep(5)
    st.rerun()
