"""Research Dashboard - Morning Brief visualization."""

import json
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Research", page_icon="📊", layout="wide")

st.title("📊 Morning Research Brief")


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
            st.error(f"Error loading initial brief: {e}")

    if pre_open_path.exists():
        try:
            with open(pre_open_path, encoding='utf-8') as f:
                briefs["pre_open"] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading pre-open brief: {e}")

    return briefs


# Date selector
col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input("Select Date", value=datetime.now().date())
with col2:
    brief_type = st.selectbox("Brief Type", ["pre_open", "initial"])

# Load briefs
briefs_dir = Path(__file__).parent.parent.parent.parent / "data" / "research" / "briefs"
briefs = load_briefs(briefs_dir, datetime.combine(selected_date, datetime.min.time()))

if brief_type in briefs:
    brief = briefs[brief_type]

    # Market Regime
    regime = brief.get("market_regime", {})
    state_emoji = {"risk-on": "🟢", "risk-off": "🔴", "neutral": "🟡"}.get(regime.get("state", ""), "⚪")

    st.markdown(f"""
    ### Market Regime {state_emoji}
    **State:** {regime.get('state', 'N/A')} | **Trend:** {regime.get('trend', 'N/A')}

    {regime.get('summary', 'No summary available')}
    """)

    st.divider()

    # Top Ideas
    st.subheader("🎯 Top Ideas")

    ideas = brief.get("ideas", [])
    if ideas:
        for idea in ideas:
            direction_emoji = "⬆️" if idea.get("direction") == "LONG" else "⬇️"

            with st.expander(
                f"#{idea.get('rank', '?')} {idea.get('ticker', 'N/A')} {idea.get('direction', '')} {direction_emoji} "
                f"[{idea.get('conviction', 'N/A')}] - {idea.get('position_size', 'N/A')} Position"
            ):
                st.markdown(f"""
                **Catalyst:** {idea.get('catalyst', 'N/A')}

                **Thesis:** {idea.get('thesis', 'N/A')}

                **Technical Levels:**
                - Support: ${idea.get('technical', {}).get('support', 'N/A')}
                - Resistance: ${idea.get('technical', {}).get('resistance', 'N/A')}
                - Entry Zone: {idea.get('technical', {}).get('entry_zone', 'N/A')}

                **Risk/Reward:**
                - Entry: ${idea.get('risk_reward', {}).get('entry', 'N/A')}
                - Stop: ${idea.get('risk_reward', {}).get('stop', 'N/A')}
                - Target: ${idea.get('risk_reward', {}).get('target', 'N/A')}
                - R:R: {idea.get('risk_reward', {}).get('ratio', 'N/A')}

                **Kill Switch:** {idea.get('kill_switch', 'N/A')}
                """)
    else:
        st.info("No trading ideas for this date")

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
            st.info("No watchlist items")

    with col2:
        st.subheader("⚠️ Risks")
        risks = brief.get("risks", [])
        if risks:
            for risk in risks:
                st.markdown(f"• {risk}")
        else:
            st.info("No risks identified")

    st.divider()

    # Key Questions
    st.subheader("❓ Key Questions")
    questions = brief.get("key_questions", [])
    if questions:
        for q in questions:
            st.markdown(f"• {q}")
    else:
        st.info("No key questions")

    # Metadata
    st.divider()
    st.caption(
        f"Generated: {brief.get('generated_at', 'N/A')} | "
        f"Fetch: {brief.get('fetch_duration_seconds', 0):.1f}s | "
        f"Analysis: {brief.get('analysis_duration_seconds', 0):.1f}s | "
        f"Sources: {', '.join(brief.get('data_sources_used', []))}"
    )

else:
    st.warning(f"No {brief_type} brief found for {selected_date}")
    st.info("Briefs are generated at 12:00 (initial) and 15:00 (pre_open) Almería time.")
