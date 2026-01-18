# Siguientes Pasos para App Funcional

**Estado actual:** Sistema rediseñado para usar Grok (xAI) como fuente única de señales sociales.

**Cambio principal:** Eliminamos FinTwitBERT, Stocktwits, Reddit y Twitter/twscrape. Ahora solo Grok + Alpaca.

---

## ARQUITECTURA SIMPLIFICADA

```
Grok (señales X/Twitter) → Claude (análisis) → Alpaca (validación + ejecución)
```

| Componente | Proveedor | Propósito |
|------------|-----------|-----------|
| Señales sociales | Grok (xAI) | Tweets con sentiment nativo |
| Análisis profundo | Claude | Catalizadores y riesgo |
| Datos de mercado | Alpaca | Precios, volumen, técnicos |
| Ejecución | Alpaca | Órdenes de trading |

---

## PRIMER LANZAMIENTO - Pasos Inmediatos

### Paso 1: Obtener API Keys (10 minutos)

#### Grok (xAI) - NUEVO
1. Ir a [xAI Console](https://console.x.ai/)
2. Crear cuenta y obtener API key
3. Agregar a `.env`:

```bash
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxx
```

#### Alpaca (Trading)
1. Crear cuenta en [Alpaca](https://alpaca.markets/)
2. Obtener API keys (Paper Trading)
3. Agregar a `.env`:

```bash
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_PAPER=true
```

#### Anthropic (Claude)
1. Crear cuenta en [Anthropic Console](https://console.anthropic.com/)
2. Generar API key
3. Agregar a `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

#### Telegram (Notificaciones)
1. Crear bot con [@BotFather](https://t.me/BotFather)
2. Obtener chat_id
3. Agregar a `.env`:

```bash
TELEGRAM_BOT_TOKEN=1234567890:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
```

### Paso 2: Verificar .env Completo

```bash
# .env debe tener estas 6 variables:
XAI_API_KEY=xai-...
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### Paso 3: Instalar Dependencias

```bash
uv sync
```

### Paso 4: Primer Arranque (~30 segundos)

```bash
uv run python main.py
```

**Salida esperada:**
```
============================================================
Starting Intraday Trading System
Mode: paper
Version: 2.0.0
============================================================
✓ Settings loaded from config/settings.yaml
✓ Environment variables validated
✓ Data directories verified
✓ Alpaca connected (Paper mode: True, Cash: $100,000.00)
✓ Telegram connected (sent startup message)
✓ Claude API verified
✓ GrokCollector initialized
✓ SourceProfileStore initialized
✓ DynamicCredibilityManager initialized
✓ SignalOutcomeTracker initialized
✓ Pipeline components ready
✓ TradingOrchestrator started
============================================================
Listening for signals... (Press Ctrl+C to stop)
```

**Nota:** Ya NO hay descarga de FinTwitBERT (5-10 min). Arranque inmediato.

### Paso 5: Arrancar Dashboard

```bash
# En otra terminal
uv run streamlit run src/dashboard/Home.py --server.port 8501
```

---

## CONFIGURACIÓN DE BÚSQUEDAS GROK

Edita `config/settings.yaml` para personalizar qué busca Grok:

```yaml
collectors:
  grok:
    enabled: true
    refresh_interval_seconds: 30
    search_queries:
      # Tickers que te interesan
      - "$NVDA"
      - "$TSLA"
      - "$AMD"
      - "$AAPL"
      - "$META"

      # Términos de options flow
      - "unusual options activity"
      - "dark pool"
      - "options flow"
      - "block trade"

      # Cuentas institucionales específicas
      - "from:unusual_whales"
      - "from:DeItaone"
      - "from:FirstSquawk"
      - "from:zerohedge"
```

---

## CREDIBILIDAD DINÁMICA

El sistema aprende automáticamente qué fuentes son confiables.

### Seeds Iniciales (Día 1)

```yaml
# config/settings.yaml
scoring:
  tier1_seeds:
    - unusual_whales    # 1.3x multiplier inicial
    - DeItaone          # 1.3x
    - FirstSquawk       # 1.3x
    - optionsflow       # 1.2x
```

### Aprendizaje Automático (Semana 1-2)

1. Sistema trackea cada señal y su autor
2. Después de 30 min, evalúa si fue correcta (+1% = éxito)
3. Actualiza accuracy del autor
4. Ajusta multiplier automáticamente:

| Accuracy | Multiplier |
|----------|------------|
| ≥75% | 1.3x |
| ≥60% | 1.1x |
| ≥50% | 1.0x |
| ≥40% | 0.8x |
| <40% | 0.5x |

### Ver Perfiles de Fuentes

Los perfiles se guardan en `data/sources/`:

```bash
cat data/sources/unusual_whales.json
```

```json
{
  "author_id": "unusual_whales",
  "source_type": "grok",
  "total_signals": 23,
  "correct_signals": 18,
  "accuracy": 0.78,
  "category": "options_flow",
  "first_seen": "2026-01-18T10:30:00",
  "last_seen": "2026-01-18T15:45:00"
}
```

---

## COSTOS ESTIMADOS

| Servicio | Costo Mensual |
|----------|---------------|
| Grok API | ~$60-100 |
| Claude API | ~$20-50 |
| Alpaca | Gratis (paper) |
| Telegram | Gratis |
| **Total** | **~$80-150/mes** |

vs antes: X API Pro $5,000/mes + hosting modelo = $5,100+

---

## PLAN DE VALIDACIÓN

### Primera Semana

- [ ] Sistema arranca sin errores
- [ ] Grok está recibiendo tweets
- [ ] Señales aparecen en dashboard
- [ ] Telegram envía notificaciones
- [ ] Perfiles de fuentes se crean en data/sources/

### Segunda Semana

- [ ] 50+ señales procesadas
- [ ] Sistema empieza a diferenciar fuentes confiables
- [ ] Primeros trades ejecutados en paper
- [ ] Journal registrando operaciones

### Primer Mes

- [ ] Accuracy de tier1 sources > 60%
- [ ] Sistema identificó nuevas fuentes confiables
- [ ] Profit factor > 1.0 en paper trading
- [ ] Circuit breakers funcionando correctamente

---

## CHECKLIST PRE-PRODUCCIÓN

### APIs
- [ ] XAI_API_KEY configurado
- [ ] ALPACA_API_KEY configurado
- [ ] ANTHROPIC_API_KEY configurado
- [ ] TELEGRAM_BOT_TOKEN configurado

### Sistema
- [ ] Primer arranque exitoso
- [ ] Dashboard funcional
- [ ] Notificaciones llegando a Telegram

### Trading
- [ ] 50+ operaciones en paper
- [ ] Profit factor > 1.2
- [ ] Max drawdown < 15%
- [ ] Circuit breakers probados

---

## TROUBLESHOOTING

### Error: "XAI_API_KEY not found"
- Verificar que `.env` tiene la variable
- Confirmar formato: `XAI_API_KEY=xai-...`

### Error: "Grok API rate limit"
- Aumentar `refresh_interval_seconds` en settings.yaml
- Reducir número de `search_queries`

### No aparecen señales
- Verificar horario de mercado (9:30-16:00 ET)
- Revisar que search_queries incluyen tickers activos
- Verificar logs de GrokCollector

### Señales pero no trades
- Revisar MarketGate (puede estar bloqueando)
- Verificar thresholds de scoring
- Revisar circuit breakers

---

## DOCUMENTACIÓN RELACIONADA

- [Diseño de Integración Grok](plans/2026-01-18-grok-integration-design.md)
- [Diseño del Pipeline](plans/2026-01-18-main-integration-design.md)
- [Manual de Uso](MANUAL-DE-USO.md)

---

*Documento actualizado: 2026-01-18*
