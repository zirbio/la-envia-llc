# Diseño de Integración Grok

**Fecha:** 2026-01-18
**Estado:** Aprobado
**Propósito:** Integrar Grok (xAI) como fuente única de señales sociales con credibilidad dinámica

---

## Resumen Ejecutivo

Este diseño simplifica el sistema de trading reemplazando múltiples collectors (Stocktwits, Reddit, Twitter/twscrape) y FinTwitBERT por una única integración con Grok que provee:

1. **Datos de X/Twitter en tiempo real** via `x_search`
2. **Sentiment nativo** incluido en resultados (-1 a +1)
3. **Credibilidad dinámica** que aprende qué fuentes aciertan

### Beneficios

| Aspecto | Antes | Después |
|---------|-------|---------|
| Collectors | 3 (Stocktwits, Reddit, Twitter) | 1 (Grok) |
| Modelo local | FinTwitBERT 500MB | Ninguno |
| Tiempo arranque | 5-10 min (descarga modelo) | ~10 seg |
| Credibilidad | Estática (YAML) | Dinámica (aprende) |
| Costo mensual | ~$5,000 (X API Pro) | ~$60-100 (Grok) |

---

## Arquitectura

### Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FUENTES DE DATOS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SEÑALES SOCIALES (Grok)          DATOS DE MERCADO (Alpaca)       │
│   ─────────────────────            ──────────────────────          │
│   • Tweets de traders              • Precios en tiempo real        │
│   • Options flow alerts            • Volumen                       │
│   • Breaking news                  • RSI, MACD, ADX                │
│   • Sentiment incluido             • Datos de opciones             │
│                                                                     │
│          │                                   │                      │
│          ▼                                   ▼                      │
│   ┌─────────────┐                   ┌─────────────────┐            │
│   │GrokCollector│                   │TechnicalValidator│           │
│   └──────┬──────┘                   └────────┬────────┘            │
│          │                                   │                      │
│          └──────────────┬───────────────────┘                      │
│                         ▼                                           │
│                  ┌─────────────┐                                   │
│                  │SignalScorer │  ← DynamicCredibilityManager      │
│                  └──────┬──────┘                                   │
│                         ▼                                           │
│                  ┌─────────────┐                                   │
│                  │TradeExecutor│  → Alpaca (órdenes)               │
│                  └──────┬──────┘                                   │
│                         │                                           │
│                         ▼                                           │
│              ┌───────────────────┐                                 │
│              │SignalOutcomeTracker│ → Feedback loop                │
│              └───────────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Proveedores

| Componente | Proveedor | Propósito |
|------------|-----------|-----------|
| Señales sociales | Grok (xAI) | Tweets de X/Twitter con sentiment |
| Análisis profundo | Claude (Anthropic) | Clasificación de catalizadores |
| Validación técnica | Alpaca | Datos de precio/volumen |
| Ejecución | Alpaca | Órdenes de compra/venta |

---

## Componentes Nuevos

### 1. SourceProfile

Modelo de datos para trackear cada fuente/autor:

```python
@dataclass
class SourceProfile:
    author_id: str              # username único
    source_type: SourceType     # GROK

    # Métricas de credibilidad
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0       # correct/total

    # Metadata
    followers: int | None = None
    category: str = "unknown"   # "trader", "institution", "news"

    # Timestamps
    first_seen: datetime
    last_seen: datetime

    @property
    def credibility_multiplier(self) -> float:
        if self.total_signals < 5:
            return 1.0
        if self.accuracy >= 0.75:
            return 1.3
        elif self.accuracy >= 0.60:
            return 1.1
        elif self.accuracy >= 0.50:
            return 1.0
        elif self.accuracy >= 0.40:
            return 0.8
        else:
            return 0.5
```

**Ubicación:** `src/scoring/source_profile.py`

### 2. SourceProfileStore

Persistencia de perfiles en disco (patrón igual a JournalManager):

```python
class SourceProfileStore:
    def __init__(self, data_dir: Path = Path("data/sources")):
        self._data_dir = data_dir
        self._cache: dict[str, SourceProfile] = {}

    def get(self, author_id: str) -> SourceProfile | None
    def save(self, profile: SourceProfile) -> None
    def get_or_create(self, author_id: str, source_type: SourceType) -> SourceProfile
```

**Ubicación:** `src/scoring/source_profile_store.py`

### 3. DynamicCredibilityManager

Reemplaza SourceCredibilityManager estático:

```python
class DynamicCredibilityManager:
    def __init__(
        self,
        profile_store: SourceProfileStore,
        min_signals_for_ranking: int = 5,
        tier1_sources: list[str] | None = None,  # Seeds iniciales
    ):
        ...

    def get_multiplier(self, author_id: str, source_type: SourceType) -> float
    def record_signal(self, author_id: str, source_type: SourceType, signal_id: str) -> None
    def record_outcome(self, author_id: str, was_correct: bool) -> None
```

**Ubicación:** `src/scoring/dynamic_credibility_manager.py`

### 4. GrokCollector

Collector que usa Grok API para buscar en X/Twitter:

```python
class GrokCollector(BaseCollector):
    def __init__(
        self,
        api_key: str,
        search_queries: list[str],
        refresh_interval: int = 30,
        max_results_per_query: int = 20,
    ):
        super().__init__(name="grok", source_type=SourceType.GROK)

    async def connect(self) -> None
    async def disconnect(self) -> None
    async def stream(self) -> AsyncIterator[SocialMessage]
```

**Ubicación:** `src/collectors/grok_collector.py`

### 5. SignalOutcomeTracker

Feedback loop para medir accuracy:

```python
class SignalOutcomeTracker:
    def __init__(
        self,
        credibility_manager: DynamicCredibilityManager,
        alpaca_client: AlpacaClient,
        evaluation_window_minutes: int = 30,
        success_threshold_percent: float = 1.0,
    ):
        ...

    def record_signal(self, signal_id, author_id, symbol, direction, entry_price) -> None
    async def evaluate_pending(self) -> None
```

**Ubicación:** `src/scoring/signal_outcome_tracker.py`

---

## Componentes Eliminados

| Componente | Razón |
|------------|-------|
| `SentimentAnalyzer` | Grok incluye sentiment nativo |
| `AnalyzerManager` | Simplificado, solo Claude |
| `StocktwitsCollector` | Redundante con Grok |
| `RedditCollector` | Redundante con Grok |
| `TwitterCollector` | Reemplazado por Grok |
| `SourceCredibilityManager` | Reemplazado por versión dinámica |
| FinTwitBERT model (500MB) | No necesario |

---

## Configuración

### Variables de Entorno

```bash
# .env
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_PAPER=true
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
XAI_API_KEY=xai-xxxxxxxxxxxxxxxx
TELEGRAM_BOT_TOKEN=1234567890:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
```

### settings.yaml

```yaml
# config/settings.yaml

collectors:
  grok:
    enabled: true
    refresh_interval_seconds: 30
    max_results_per_query: 20
    search_queries:
      # Tickers
      - "$NVDA"
      - "$TSLA"
      - "$AMD"
      - "$AAPL"
      - "$META"
      # Términos institucionales
      - "unusual options activity"
      - "dark pool"
      - "options flow"
      # Cuentas específicas
      - "from:unusual_whales"
      - "from:DeItaone"
      - "from:FirstSquawk"

scoring:
  # Seeds de credibilidad inicial
  tier1_seeds:
    - unusual_whales
    - DeItaone
    - FirstSquawk
    - optionsflow
    - zerohedge

  # Credibilidad dinámica
  min_signals_for_dynamic: 5
  evaluation_window_minutes: 30
  success_threshold_percent: 1.0
```

---

## Integración main.py

### Fases de Inicialización

| Fase | Antes | Después |
|------|-------|---------|
| 1. Config | Igual | + XAI_API_KEY |
| 2. Infra | Alpaca + Telegram | Igual |
| 3. Análisis | FinTwitBERT + Claude | Solo Claude |
| 4. Collectors | ST + Reddit + Twitter | Solo Grok |
| 5. Pipeline | SourceCredibilityManager | DynamicCredibilityManager + OutcomeTracker |
| 6. Orchestrator | Igual | + outcome_tracker |

### Directorios de Datos

```
data/
├── trades/          # Journal de trades
├── signals/         # Señales procesadas
├── cache/           # Cache general
├── sources/         # [NUEVO] SourceProfiles
└── backtest_results/
```

---

## Feedback Loop: Cómo Aprende el Sistema

```
1. Tweet de @trader123: "$NVDA bullish, earnings catalyst"
   ↓
2. GrokCollector captura (sentiment: 0.8 bullish)
   ↓
3. TechnicalValidator confirma (RSI ok, volume ok)
   ↓
4. SignalScorer:
   - credibility_multiplier(@trader123) = 1.0 (nuevo)
   - score = 72 (MODERATE)
   ↓
5. Trade ejecutado: BUY NVDA @ $500
   ↓
6. SignalOutcomeTracker.record_signal()
   ↓
7. [30 minutos después]
   ↓
8. evaluate_pending(): NVDA @ $512 (+2.4%)
   ↓
9. record_outcome(@trader123, was_correct=True)
   ↓
10. SourceProfile actualizado:
    - total_signals: 1 → 2
    - correct_signals: 1 → 2
    - accuracy: 100%
    - (aún no ajusta multiplier, necesita 5 señales)
   ↓
11. [Después de 5+ señales]
    - accuracy: 80%
    - credibility_multiplier: 1.3x
    ↓
12. Próxima señal de @trader123 tiene +30% más peso
```

---

## Checklist de Implementación

### Archivos Nuevos

- [ ] `src/models/social_message.py` - Añadir `SourceType.GROK`
- [ ] `src/collectors/grok_collector.py`
- [ ] `src/scoring/source_profile.py`
- [ ] `src/scoring/source_profile_store.py`
- [ ] `src/scoring/dynamic_credibility_manager.py`
- [ ] `src/scoring/signal_outcome_tracker.py`

### Archivos Modificados

- [ ] `src/config/settings.py` - GrokCollectorConfig, scoring seeds
- [ ] `config/settings.yaml` - Sección grok, seeds
- [ ] `main.py` - Nueva inicialización simplificada
- [ ] `src/orchestrator/trading_orchestrator.py` - Integrar outcome_tracker
- [ ] `src/collectors/__init__.py` - Export GrokCollector
- [ ] `src/scoring/__init__.py` - Exports nuevos

### Archivos Eliminados/Deprecados

- [ ] Marcar como deprecated: `src/analyzers/sentiment_analyzer.py`
- [ ] Marcar como deprecated: `src/analyzers/analyzer_manager.py`
- [ ] Marcar como deprecated: `src/collectors/stocktwits_collector.py`
- [ ] Marcar como deprecated: `src/collectors/reddit_collector.py`
- [ ] Marcar como deprecated: `src/collectors/twitter_collector.py`
- [ ] Marcar como deprecated: `src/scoring/source_credibility_manager.py`

### Tests

- [ ] `tests/collectors/test_grok_collector.py`
- [ ] `tests/scoring/test_source_profile.py`
- [ ] `tests/scoring/test_source_profile_store.py`
- [ ] `tests/scoring/test_dynamic_credibility_manager.py`
- [ ] `tests/scoring/test_signal_outcome_tracker.py`
- [ ] `tests/integration/test_grok_pipeline.py`

---

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Grok API no disponible | Retry con backoff exponencial |
| Rate limiting de Grok | Configurar refresh_interval apropiado |
| X/Twitter cambia formato | Parseo flexible con fallbacks |
| Seeds incorrectos | Sistema aprende y corrige en ~2 semanas |

---

## Métricas de Éxito

1. **Arranque < 30 segundos** (vs 5-10 min con FinTwitBERT)
2. **Accuracy promedio de fuentes tier1 > 60%** después de 2 semanas
3. **Sistema identifica automáticamente** nuevas fuentes confiables
4. **Costo < $150/mes** (Grok + Claude + Alpaca)

---

## Próximos Pasos

1. Obtener API key de xAI (Grok)
2. Implementar GrokCollector
3. Implementar sistema de credibilidad dinámica
4. Actualizar main.py
5. Testing en paper trading
6. Monitorear accuracy de fuentes

---

*Diseño aprobado: 2026-01-18*
