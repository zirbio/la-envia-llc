"""Analysis page - Performance metrics and charts."""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Análisis | Trading Dashboard", page_icon="📊", layout="wide")

st.title("📊 Análisis de Rendimiento")

col1, col2 = st.columns([1, 3])

with col1:
    period = st.selectbox("Período", options=["Hoy", "Semana", "Mes", "Custom"], index=1)

with col2:
    if period == "Custom":
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Desde", value=date.today() - timedelta(days=30))
        with col_end:
            end_date = st.date_input("Hasta", value=date.today())
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days={"Hoy": 0, "Semana": 7, "Mes": 30}[period])

st.divider()

st.subheader("📈 Métricas Principales")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Win Rate", "65%", delta="+5%")
with col2:
    st.metric("Profit Factor", "2.1", delta="+0.3")
with col3:
    st.metric("Expectancy", "0.85R", delta="+0.1R")
with col4:
    st.metric("Total P&L", "$1,250", delta="+$350")

st.divider()

st.subheader("📉 Gráficos")

tab1, tab2, tab3 = st.tabs(["Equity Curve", "P&L por Día", "Win Rate por Hora"])

with tab1:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    equity = [10000 + i * 50 for i in range(len(dates))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", name="Equity", line=dict(color="green")))
    fig.update_layout(title="Equity Curve", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    pnl_data = {"Día": ["Lun", "Mar", "Mié", "Jue", "Vie"], "P&L": [150, -50, 200, 100, -30]}
    df = pd.DataFrame(pnl_data)
    colors = ["green" if x >= 0 else "red" for x in df["P&L"]]
    fig = go.Figure(data=[go.Bar(x=df["Día"], y=df["P&L"], marker_color=colors)])
    fig.update_layout(title="P&L por Día", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    hours = list(range(9, 16))
    win_rates = [0.6, 0.7, 0.65, 0.55, 0.7, 0.75, 0.5]
    fig = go.Figure(data=[go.Bar(x=hours, y=win_rates, marker_color="blue")])
    fig.update_layout(title="Win Rate por Hora", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("🔍 Patrones Identificados")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Mejores:**")
    st.success("⏰ Mejor hora: 10:00 AM (75% WR)")
    st.success("📊 Mejor símbolo: NVDA (+$450)")

with col2:
    st.markdown("**A Mejorar:**")
    st.warning("⏰ Peor hora: 3:00 PM (45% WR)")
    st.warning("📊 Peor símbolo: TSLA (-$120)")
