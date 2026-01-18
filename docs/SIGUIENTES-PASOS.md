# Siguientes Pasos para App Funcional

**Estado actual:** Sistema completo con 12 fases implementadas y 664 tests pasando.

**Lo que falta:** Configuración de APIs externas, integración de componentes en main.py, y pruebas en paper trading.

---

## 1. Configuración de APIs Externas

### 1.1 Alpaca (Trading)

1. Crear cuenta en [Alpaca](https://alpaca.markets/)
2. Obtener API keys desde el dashboard (Paper Trading primero)
3. Configurar en `.env`:

```bash
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_PAPER=true
```

### 1.2 Anthropic (Claude API)

1. Crear cuenta en [Anthropic Console](https://console.anthropic.com/)
2. Generar API key
3. Agregar a `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXX
```

### 1.3 Telegram (Notificaciones)

1. Crear bot con [@BotFather](https://t.me/BotFather):
   - Enviar `/newbot`
   - Elegir nombre y username
   - Guardar el token
2. Obtener tu chat_id enviando mensaje al bot y visitando:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Agregar a `.env`:

```bash
TELEGRAM_BOT_TOKEN=1234567890:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TELEGRAM_CHAT_ID=123456789
```

### 1.4 Reddit API (Opcional)

1. Ir a [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Crear "script" application
3. Agregar a `.env`:

```bash
REDDIT_CLIENT_ID=XXXXXXXXXXXXXX
REDDIT_CLIENT_SECRET=XXXXXXXXXXXXXXXXXXXXXX
REDDIT_USER_AGENT=TradingBot/1.0 by u/tu_usuario
```

### 1.5 Twitter/X (Opcional)

El sistema usa `twscrape` que no requiere API oficial. Sin embargo, necesita cuentas de Twitter configuradas:

```python
# scripts/setup_twitter_accounts.py
from twscrape import AccountsPool

pool = AccountsPool()
await pool.add_account("user1", "pass1", "email1", "email_pass1")
await pool.login_all()
```

---

## 2. Actualizar main.py con Pipeline Completo

✅ **COMPLETADO** - El `main.py` ahora integra todos los componentes del sistema.

### Cambios Implementados

1. **Inicialización Secuencial en 6 Fases:**
   - Fase 1: Configuración (env vars, settings, data dirs)
   - Fase 2: Infraestructura (Alpaca, Telegram)
   - Fase 3: Análisis (FinTwitBERT, Claude)
   - Fase 4: Collectors (Stocktwits, Reddit opcional)
   - Fase 5: Pipeline (Scorer, Validator, Gate, Risk, Journal, Executor)
   - Fase 6: Orchestrator (coordinador principal)

2. **Funciones Helper:**
   - `validate_env_vars()` - Valida variables requeridas
   - `create_data_dirs()` - Crea estructura de directorios
   - `print_startup_banner()` - Banner de inicio

3. **Manejo de Errores:**
   - Fail-fast: El sistema se detiene inmediatamente si falla un componente crítico
   - Mensajes claros indicando qué falló y cómo arreglarlo
   - Reddit opcional: Se salta si no hay credenciales (warning, no error)

4. **Shutdown Graceful:**
   - Detiene orchestrator procesando mensajes pendientes
   - Desconecta Alpaca
   - Envía notificación final a Telegram
   - Logs claros de cada paso

5. **Observabilidad:**
   - Logging estructurado con timestamps
   - Checkmark ✓ para cada componente inicializado
   - Mensajes informativos durante operación

### Cómo Ejecutar

```bash
# Crear directorios de datos (primera vez)
mkdir -p data/trades data/signals data/cache data/backtest_results

# Ejecutar sistema
uv run python main.py
```

### Salida Esperada

Ver `docs/plans/2026-01-18-main-integration-design.md` sección "Execution" para la salida completa esperada.

---

## 3. Crear Archivos de Configuración Faltantes

### 3.1 config/risk_params.yaml

```yaml
levels:
  level_1:
    name: "Conservative"
    max_risk_per_trade: 0.5
    max_position_size: 2.5
    max_concurrent_positions: 2

  level_2:
    name: "Standard"
    max_risk_per_trade: 1.0
    max_position_size: 5.0
    max_concurrent_positions: 3
```

### 3.2 config/social_sources.yaml

```yaml
twitter:
  smart_money:
    accounts:
      - username: "unusual_whales"
        weight: 1.0
      - username: "OptionsHawk"
        weight: 0.9

reddit:
  subreddits:
    - name: "wallstreetbets"
      weight: 0.7
    - name: "stocks"
      weight: 0.9
```

---

## 4. Crear Directorios de Datos

```bash
mkdir -p data/trades
mkdir -p data/signals
mkdir -p data/cache
mkdir -p data/backtest_results
```

---

## 5. Descargar Modelo FinTwitBERT

Primera ejecución descarga automáticamente, pero para pre-cargar:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "StephanAkkerman/FinTwitBERT-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

## 6. Ejecutar Dashboard (Streamlit)

El dashboard necesita un archivo de entrada:

```bash
uv run streamlit run src/dashboard/Home.py --server.port 8501
```

---

## 7. Orden de Pruebas Recomendado

### Fase 1: Verificar Conexiones (1-2 días)

```bash
# 1. Verificar Alpaca
python -c "
from src.execution import AlpacaClient
import asyncio

async def test():
    client = AlpacaClient(paper=True)
    await client.connect()
    account = await client.get_account()
    print(f'Cash: \${account[\"cash\"]:,.2f}')
    await client.disconnect()

asyncio.run(test())
"

# 2. Verificar Telegram
python -c "
from src.notifications import TelegramNotifier
import asyncio

async def test():
    notifier = TelegramNotifier()
    await notifier.send_alert('Test de conexión exitoso!')

asyncio.run(test())
"

# 3. Verificar Claude
python -c "
from src.analyzers import ClaudeAnalyzer
import asyncio

async def test():
    analyzer = ClaudeAnalyzer()
    result = await analyzer.analyze_message('NVDA looking bullish')
    print(result)

asyncio.run(test())
"
```

### Fase 2: Probar Pipeline Parcial (2-3 días)

1. Ejecutar collectors solos y verificar mensajes
2. Pasar mensajes al analyzer y verificar sentiment
3. Probar scoring con datos mock
4. Verificar gate decisions

### Fase 3: Paper Trading Completo (1-2 semanas)

1. Ejecutar sistema completo en modo paper
2. Monitorear señales generadas
3. Verificar journal entries
4. Ajustar thresholds según resultados

### Fase 4: Revisión y Ajustes (1 semana)

1. Analizar métricas del journal
2. Ajustar pesos del scoring
3. Refinar filtros de calidad
4. Documentar lecciones aprendidas

---

## 8. Checklist Pre-Producción

- [ ] `.env` configurado con todas las APIs
- [ ] Conexión Alpaca verificada
- [ ] Conexión Telegram verificada
- [ ] Modelo FinTwitBERT descargado
- [ ] Directorios de datos creados
- [ ] main.py actualizado con pipeline completo
- [ ] Dashboard funcional
- [ ] 50+ trades en paper trading
- [ ] Profit factor > 1.2 en paper
- [ ] Max drawdown < 15% en paper
- [ ] Circuit breakers probados manualmente

---

## 9. Consideraciones de Seguridad

1. **Nunca** subir `.env` a git
2. **Siempre** empezar en paper trading
3. **Nunca** desactivar circuit breakers
4. Mantener logs de todas las operaciones
5. Revisar journal semanalmente

---

## 10. Recursos Adicionales

- [Alpaca Docs](https://docs.alpaca.markets/)
- [Anthropic Docs](https://docs.anthropic.com/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [FinTwitBERT Model](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment)
- [Streamlit Docs](https://docs.streamlit.io/)

---

*Documento generado: 2026-01-17*
