# Configuración Optimizada de Costos - Grok Integration

**Fecha:** 2026-01-18
**Objetivo:** Minimizar costos durante fase de prueba (~$40-50/mes)

---

## Configuración Ultra-Optimizada

### Cambios Clave

| Parámetro | Valor Original | Valor Optimizado | Reducción |
|-----------|---------------|------------------|-----------|
| `refresh_interval` | 30 seg | **900 seg (15 min)** | 96% menos búsquedas |
| `max_results_per_query` | 20 | **5** | 75% menos resultados |
| `search_queries` | 10+ keywords | **3 cuentas tier1** | 70% menos queries |

### Búsquedas Activas

Solo **3 cuentas verificadas de alto valor**:
- `from:unusual_whales` - Options flow institucional
- `from:DeItaone` - Breaking news financiero
- `from:FirstSquawk` - Noticias mercado

**Estrategia:** Evitar búsquedas amplias por keywords ($NVDA, "options flow", etc.) que generan muchos resultados. Solo seguir cuentas tier1 verificadas.

---

## Estimado de Costos Mensual

### Búsquedas X (x_search)

```
Refresh cada 15 min = 96 búsquedas/día
3 queries por búsqueda = 288 búsquedas/día
× 30 días = 8,640 búsquedas/mes

Costo: 8,640 / 1,000 × $5 = $43.20/mes
```

### Grok 4.1 Fast (procesamiento)

```
Input tokens:
  288 búsquedas/día × 5 results × 200 tokens avg = 288k tokens/día
  × 30 días = 8.64M tokens/mes
  × $0.20/1M = $1.73/mes

Output tokens (sentiment + análisis):
  8.64M × 0.3 ratio × $0.50/1M = $1.30/mes

Total Grok tokens: ~$3/mes
```

### Claude Sonnet (análisis profundo)

```
~100 señales/mes × 500 tokens = 50k tokens/mes
× $3/1M input + $15/1M output ≈ $1-2/mes
```

### **TOTAL ESTIMADO: $47-48/mes**

---

## Comparación con Plan Original

| Concepto | Plan Original | Plan Optimizado | Ahorro |
|----------|---------------|-----------------|--------|
| Búsquedas/día | 2,880 | 288 | 90% |
| Costo x_search | $432/mes | $43/mes | $389/mes |
| Costo Grok tokens | $30/mes | $3/mes | $27/mes |
| **TOTAL** | **$460-480/mes** | **$47-48/mes** | **$412-432/mes** |

---

## Camino de Escalamiento

Cuando veas señales de valor, puedes aumentar gradualmente:

### Fase 1: Prueba (actual)
- **Costo:** ~$48/mes
- Refresh: 15 min
- 3 queries (cuentas tier1)
- 5 results

### Fase 2: Expansión Moderada (+$30/mes)
```yaml
refresh_interval_seconds: 300  # 5 min
max_results_per_query: 10
search_queries:
  - from:unusual_whales
  - from:DeItaone
  - from:FirstSquawk
  - from:optionsflow        # +1 cuenta
  - $NVDA                   # +1 keyword importante
  - $TSLA                   # +1 keyword importante
```
**Costo estimado:** ~$78/mes

### Fase 3: Full Production (+$120/mes)
```yaml
refresh_interval_seconds: 120  # 2 min
max_results_per_query: 15
search_queries: [10-12 queries mixtos]
```
**Costo estimado:** ~$198/mes

---

## Qué Esperar con Configuración Optimizada

### Ventajas ✅
- Costo ultra-bajo para probar concepto
- Señales de alta calidad (solo tier1)
- Sin ruido de cuentas no verificadas
- Arranque rápido (~10 seg vs 5-10 min)

### Limitaciones ⚠️
- Solo 15-20 tweets/día procesados (vs 100-200)
- Latencia de 15 min (puede perder señales urgentes)
- Solo 3 fuentes (menor diversidad)

### Cuándo Escalar
Si después de 2-3 semanas ves:
- Accuracy de fuentes > 65%
- Señales rentables consistentes
- Sistema identifica oportunidades reales
→ Entonces vale la pena invertir más

---

## Archivos Modificados

- ✅ `config/settings.yaml` - Configuración Grok optimizada
- ✅ `.env.example` - Variable XAI_API_KEY añadida

---

## Próximo Paso

**Obtener API key de xAI:**
1. Visitar: https://console.x.ai/
2. Crear cuenta / login
3. Generar API key
4. Añadir a `.env`: `XAI_API_KEY=xai-...`

**Modelo a solicitar:** Grok 4.1 Fast + X Search access

---

*Optimización para fase de prueba - 2026-01-18*
