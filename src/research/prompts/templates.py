# src/research/prompts/templates.py

"""Prompt templates for Morning Research Agent."""

SYSTEM_PROMPT = """Eres un analista senior de equity research en un hedge fund cuantitativo.
Tu trabajo: producir el Morning Brief antes de la apertura del mercado de EE.UU.

RESPONDE SIEMPRE EN ESPAÑOL.

PRINCIPIOS:
1. Cada conclusión debe citar evidencia (fuente de datos + números específicos)
2. Distinguir HECHOS (datos) de INFERENCIAS (tu análisis)
3. Piensa como contrarian: ¿qué está perdiendo o mal valorando el mercado?
4. Expresa convicción en probabilidades, no certezas
5. Enfócate en riesgo/recompensa asimétrico (mínimo 2:1)
6. Si los datos son insuficientes, di "DATOS INSUFICIENTES" - nunca adivines

CRITERIOS DE RECHAZO (NO automático):
- Sin catalizador claro en las próximas 24-48 horas
- Riesgo/recompensa por debajo de 1.5:1
- Señales conflictivas sin resolución
- Baja liquidez (< 500K volumen promedio)"""


def build_context(data: dict) -> str:
    """Build context string from fetched data.

    Args:
        data: Dictionary with fetched market data.

    Returns:
        Formatted context string for the prompt.
    """
    futures = data.get("futures", {})
    gappers = data.get("gappers", [])
    earnings = data.get("earnings", [])
    economic = data.get("economic_events", [])
    sec = data.get("sec_filings", {"8k": [], "form4": []})
    social = data.get("social_intelligence", "Sin datos sociales disponibles")
    news = data.get("news", [])

    # Format gappers table
    gappers_table = ""
    for g in gappers:
        gappers_table += f"| {g.get('ticker', 'N/A')} | {g.get('gap_percent', 0):.1f}% | {g.get('volume', 0):,} | {g.get('catalyst', 'Desconocido')} |\n"
    if not gappers_table:
        gappers_table = "| No se encontraron gappers significativos |\n"

    return f"""<context>
<market_snapshot>
Fecha: {data.get('date', 'Desconocido')} | Hora: {data.get('time', 'Desconocido')} ET
Futuros ES: {futures.get('es', 'N/A')} ({futures.get('es_change', 0):.1f}%)
Futuros NQ: {futures.get('nq', 'N/A')} ({futures.get('nq_change', 0):.1f}%)
VIX: {data.get('vix', 'N/A')}
</market_snapshot>

<premarket_gappers>
| Ticker | Gap% | Volumen | Catalizador |
|--------|------|---------|-------------|
{gappers_table}</premarket_gappers>

<earnings_hoy>
{', '.join(earnings) if earnings else 'Sin earnings importantes hoy'}
</earnings_hoy>

<calendario_economico>
{chr(10).join(economic) if economic else 'Sin eventos importantes programados'}
</calendario_economico>

<filings_sec_24h>
8-K Eventos Materiales: {len(sec.get('8k', []))} filings
Form 4 Actividad Insiders: {len(sec.get('form4', []))} filings
</filings_sec_24h>

<inteligencia_social>
{social}
</inteligencia_social>

<noticias_overnight>
{chr(10).join(news) if news else 'Sin noticias significativas overnight'}
</noticias_overnight>
</context>"""


TASK_PROMPT = """Analiza TODOS los datos anteriores. Produce el MORNING BRIEF de hoy.

IMPORTANTE: Responde en ESPAÑOL.

SECCIONES REQUERIDAS:

1. RÉGIMEN DE MERCADO (3-4 oraciones)
   - Estado actual del mercado (risk-on/off, tendencia/rango)
   - Desarrollos clave overnight
   - Qué señalan el VIX y los futuros

2. IDEAS PRINCIPALES (máx 5, ordenadas por convicción)
   Para CADA idea proporciona:
   - Ticker, Dirección (LONG/SHORT), Convicción (HIGH/MED/LOW)
   - Catalizador: ¿Qué impulsa esto HOY? (evento/dato específico)
   - Tesis: ¿Por qué se moverá el precio en tu dirección?
   - Técnico: Niveles clave (soporte, resistencia, zona de entrada)
   - Riesgo/Recompensa: Entrada, Stop, Objetivo (con ratio R:R)
   - Tamaño de Posición: FULL (1x) / HALF (0.5x) / QUARTER (0.25x)
   - Kill Switch: ¿Qué invalidaría esta idea?

3. WATCHLIST (3-5 tickers)
   - Configurándose pero no accionables hoy
   - ¿Qué trigger los haría operables?

4. RIESGOS Y MINAS
   - Eventos que podrían destruir posiciones
   - Horarios programados para tener cuidado

5. PREGUNTAS CLAVE
   - ¿Qué información cambiaría tu tesis hoy?

Devuelve SOLO JSON válido siguiendo este schema:
{
  "market_regime": {
    "state": "risk-on|risk-off|neutral",
    "trend": "bullish|bearish|ranging",
    "summary": "string en español"
  },
  "ideas": [
    {
      "rank": 1,
      "ticker": "NVDA",
      "direction": "LONG",
      "conviction": "HIGH",
      "catalyst": "string en español",
      "thesis": "string en español",
      "technical": {
        "support": 135.50,
        "resistance": 142.00,
        "entry_zone": [136.00, 137.50]
      },
      "risk_reward": {
        "entry": 137.00,
        "stop": 134.50,
        "target": 145.00,
        "ratio": "3.2:1"
      },
      "position_size": "FULL",
      "kill_switch": "string en español"
    }
  ],
  "watchlist": [
    {"ticker": "MSFT", "setup": "string en español", "trigger": "string en español"}
  ],
  "risks": ["string en español"],
  "key_questions": ["string en español"]
}"""
