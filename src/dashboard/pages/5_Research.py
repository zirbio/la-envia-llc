"""Research Dashboard - Morning Brief visualization and generation."""

import asyncio
import json
import os
import streamlit as st
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Research", page_icon="📊", layout="wide")

st.title("📊 Morning Research Brief")


def get_briefs_dir() -> Path:
    """Get the briefs directory path."""
    return Path(__file__).parent.parent.parent.parent / "data" / "research" / "briefs"


def load_briefs(briefs_dir: Path, date: datetime) -> dict:
    """Load briefs for a specific date."""
    date_str = date.strftime("%Y-%m-%d")
    briefs = {}

    initial_path = briefs_dir / f"{date_str}_initial.json"
    pre_open_path = briefs_dir / f"{date_str}_pre_open.json"

    if initial_path.exists():
        try:
            with open(initial_path, encoding='utf-8') as f:
                briefs["initial"] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error cargando brief inicial: {e}")

    if pre_open_path.exists():
        try:
            with open(pre_open_path, encoding='utf-8') as f:
                briefs["pre_open"] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error cargando brief pre-open: {e}")

    return briefs


def save_brief(brief, brief_type: str) -> Path:
    """Save a generated brief to disk.

    Args:
        brief: DailyBrief instance.
        brief_type: Type of brief (initial or pre_open).

    Returns:
        Path where the brief was saved.
    """
    briefs_dir = get_briefs_dir()
    briefs_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = briefs_dir / f"{date_str}_{brief_type}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(brief.model_dump(mode="json"), f, indent=2, default=str)

    return filepath


def display_brief(brief: dict):
    """Display a brief in the UI.

    Args:
        brief: Brief data as a dictionary.
    """
    # Market Regime
    regime = brief.get("market_regime", {})
    state_emoji = {"risk-on": "🟢", "risk-off": "🔴", "neutral": "🟡"}.get(regime.get("state", ""), "⚪")

    st.markdown(f"""
    ### Régimen de Mercado {state_emoji}
    **Estado:** {regime.get('state', 'N/A')} | **Tendencia:** {regime.get('trend', 'N/A')}

    {regime.get('summary', 'Sin resumen disponible')}
    """)

    st.divider()

    # Top Ideas
    st.subheader("🎯 Ideas Principales")

    ideas = brief.get("ideas", [])
    if ideas:
        for idea in ideas:
            direction_emoji = "⬆️" if idea.get("direction") == "LONG" else "⬇️"

            with st.expander(
                f"#{idea.get('rank', '?')} {idea.get('ticker', 'N/A')} {idea.get('direction', '')} {direction_emoji} "
                f"[{idea.get('conviction', 'N/A')}] - Posición {idea.get('position_size', 'N/A')}"
            ):
                st.markdown(f"""
                **Catalizador:** {idea.get('catalyst', 'N/A')}

                **Tesis:** {idea.get('thesis', 'N/A')}

                **Niveles Técnicos:**
                - Soporte: ${idea.get('technical', {}).get('support', 'N/A')}
                - Resistencia: ${idea.get('technical', {}).get('resistance', 'N/A')}
                - Zona de Entrada: {idea.get('technical', {}).get('entry_zone', 'N/A')}

                **Riesgo/Beneficio:**
                - Entrada: ${idea.get('risk_reward', {}).get('entry', 'N/A')}
                - Stop: ${idea.get('risk_reward', {}).get('stop', 'N/A')}
                - Objetivo: ${idea.get('risk_reward', {}).get('target', 'N/A')}
                - R:R: {idea.get('risk_reward', {}).get('ratio', 'N/A')}

                **Kill Switch:** {idea.get('kill_switch', 'N/A')}
                """)
    else:
        st.info("No hay ideas de trading para esta fecha")

    st.divider()

    # Watchlist and Risks
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👀 Watchlist")
        watchlist = brief.get("watchlist", [])
        if watchlist:
            for item in watchlist:
                st.markdown(f"**{item.get('ticker', 'N/A')}** - {item.get('setup', '')}")
                st.caption(f"Trigger: {item.get('trigger', 'N/A')}")
        else:
            st.info("Sin elementos en watchlist")

    with col2:
        st.subheader("⚠️ Riesgos")
        risks = brief.get("risks", [])
        if risks:
            for risk in risks:
                st.markdown(f"• {risk}")
        else:
            st.info("Sin riesgos identificados")

    st.divider()

    # Key Questions
    st.subheader("❓ Preguntas Clave")
    questions = brief.get("key_questions", [])
    if questions:
        for q in questions:
            st.markdown(f"• {q}")
    else:
        st.info("Sin preguntas clave")

    # Metadata
    st.divider()
    st.caption(
        f"Generado: {brief.get('generated_at', 'N/A')} | "
        f"Fetch: {brief.get('fetch_duration_seconds', 0):.1f}s | "
        f"Análisis: {brief.get('analysis_duration_seconds', 0):.1f}s | "
        f"Fuentes: {', '.join(brief.get('data_sources_used', []))}"
    )


# =============================================================================
# Generate New Brief Section
# =============================================================================
st.subheader("Generar Nuevo Brief")

gen_col1, gen_col2 = st.columns([2, 1])
with gen_col1:
    gen_brief_type = st.selectbox(
        "Tipo de brief",
        options=["initial", "pre_open"],
        format_func=lambda x: "Initial (12:00)" if x == "initial" else "Pre-Open (15:00)",
        key="gen_brief_type"
    )
with gen_col2:
    generate_btn = st.button("🚀 Generar Brief", type="primary")

if generate_btn:
    # Check for required API keys
    grok_key = os.getenv("XAI_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")

    if not grok_key or not claude_key:
        st.error("Faltan API keys. Asegúrate de tener XAI_API_KEY y ANTHROPIC_API_KEY configurados en el archivo .env")
    else:
        with st.spinner(f"Generando brief {gen_brief_type}... (esto puede tardar 30-60 segundos)"):
            try:
                from src.research.morning_agent import MorningResearchAgent

                agent = MorningResearchAgent(
                    grok_api_key=grok_key,
                    claude_api_key=claude_key,
                )

                # Run async code
                brief = asyncio.run(agent.generate_brief(brief_type=gen_brief_type))

                # Save brief
                filepath = save_brief(brief, gen_brief_type)

                total_duration = brief.fetch_duration_seconds + brief.analysis_duration_seconds
                st.success(
                    f"Brief generado en {total_duration:.1f}s "
                    f"(fetch: {brief.fetch_duration_seconds:.1f}s, análisis: {brief.analysis_duration_seconds:.1f}s)"
                )
                st.caption(f"Guardado en: {filepath}")

                # Display the generated brief
                st.divider()
                st.subheader("Brief Generado")
                display_brief(brief.model_dump(mode="json"))

            except Exception as e:
                st.error(f"Error generando brief: {e}")
                import traceback
                st.code(traceback.format_exc())

st.divider()

# =============================================================================
# Saved Briefs Section
# =============================================================================
st.subheader("Briefs Guardados")

# Date selector
col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input("Seleccionar Fecha", value=datetime.now().date())
with col2:
    brief_type = st.selectbox("Tipo de Brief", ["pre_open", "initial"], key="view_brief_type")

# Load briefs
briefs_dir = get_briefs_dir()
briefs = load_briefs(briefs_dir, datetime.combine(selected_date, datetime.min.time()))

if brief_type in briefs:
    display_brief(briefs[brief_type])
else:
    st.warning(f"No hay brief {brief_type} para {selected_date}")
    st.info("Los briefs se generan a las 12:00 (initial) y 15:00 (pre_open) hora de Madrid.")
