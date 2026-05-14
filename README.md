# 🧬 Optimizador de Portafolio Cripto — Algoritmo Genético
**Curso:** Inteligencia Artificial (SI404) — TB1 2026

---

## Descripción del Proyecto

Sistema de optimización de portafolios de criptomonedas mediante **Algoritmo Genético (AG)**.

Dado que los precios de las criptomonedas no son predecibles con certeza, el enfoque correcto es optimizar la **distribución de capital** entre activos para maximizar el rendimiento ajustado al riesgo (Ratio de Sharpe) usando datos históricos reales. El sistema soporta dos modos de inversión:

- **Solo LONG**: el AG busca la mejor combinación de compras (posiciones alcistas).
- **LONG + SHORT**: el AG puede asignar posiciones negativas (short selling) a activos cuya caída histórica mejore el Sharpe del portafolio.

---

## Estructura del Proyecto

```
crypto_portfolio_optimizer/
├── main_gui.py           # Interfaz gráfica principal (Tkinter)
├── genetic_algorithm.py  # Implementación completa del AG
├── data_manager.py       # Datos: CoinGecko API + caché SQLite
├── crypto_cache.db       # Base de datos local (se genera automáticamente)
├── requirements.txt      # Dependencias externas
└── README.md             # Este archivo
```

---

## Instalación y Ejecución

```bash
# 1. Instalar dependencias (usar py en Windows si python no está en PATH)
py -m pip install numpy matplotlib requests

# 2. Ejecutar la aplicación
py main_gui.py
```

> **Nota:** No se requiere clave de API. CoinGecko ofrece acceso gratuito al endpoint de precios históricos. Si no hay conexión a internet, el sistema usa automáticamente la caché local o un generador de datos simulados.

---

## Cómo Funciona el Algoritmo Genético

### Representación del Individuo (Cromosoma)

Cada individuo es un vector de pesos con signo **w = [w₁, w₂, ..., wₙ] ∈ ℝᴺ** donde:

| Valor de wᵢ | Significado |
|-------------|-------------|
| wᵢ > 0 | Posición **larga** (LONG): se gana si el precio sube |
| wᵢ < 0 | Posición **corta** (SHORT): se gana si el precio baja |

**Restricciones garantizadas en todo individuo:**

- `sum(|wᵢ|) = 1` — exposición bruta = 100% del capital, sin apalancamiento
- `mn ≤ |wᵢ| ≤ mx` — cada posición activa entre el mínimo y el máximo configurados
- `sum(|wᵢ| para wᵢ < 0) ≤ short_cap` — límite total de exposición short

### Función de Fitness: Ratio de Sharpe

```
Sharpe(w) = (μₚ − Rƒ) / σₚ × √252
```

| Variable | Descripción |
|----------|-------------|
| μₚ | Retorno diario promedio: `mean(returns_matrix @ w)` |
| Rƒ | Tasa libre de riesgo diaria: `0.0001` (0.01%) |
| σₚ | Desviación estándar diaria: `std(returns_matrix @ w)` |
| √252 | Factor de anualización (252 días de trading por año) |

Un Sharpe más alto indica mejor relación retorno/riesgo. El AG maximiza esta métrica.

### Ciclo Evolutivo

```
Inicialización (Dirichlet)
        ↓
Evaluación (Sharpe Ratio)
        ↓
┌─── Selección (Torneo k=3) ───┐
│    Cruce (BLX-alpha, α=0.5)  │  × N generaciones
│    Mutación creep (gauss)     │
│    Elitismo (top 8-10)        │
└──────────────────────────────┘
        ↓
Portafolio óptimo w*
```

### Operadores Genéticos

**Selección — Torneo binario (k=3):**
Se comparan 3 individuos elegidos al azar y avanza el de mayor Sharpe. Equilibra presión selectiva y diversidad genética.

**Cruce — BLX-alpha (α=0.5):**
Genera hijos en el rango extendido `[lo − α·rng, hi + α·rng]` de los padres, permitiendo exploración más allá de los valores parentales.

**Mutación — Creep gaussiana:**
Cada gen recibe ruido `N(0, 0.03)` con probabilidad 0.30, equivalente a una perturbación de ±3% del capital. Preserva el signo y la magnitud aproximada del gen original, evitando saltos destructivos. En modo LONG+SHORT, el cambio de dirección (flip de signo) solo se aplica a activos de magnitud pequeña (≤ 3×min_weight) con probabilidad 0.10.

**Normalización — `_enforce_constraints`:**
Aplicada tras cada operación genética. Trabaja sobre magnitudes (`abs`) y restaura los signos al final, garantizando que `sum(|w|) = 1` y todos los límites se cumplan exactamente en cada individuo evaluado.

---

## Parámetros Configurables en la GUI

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| Tamaño de Población | 80 | Individuos por generación |
| Generaciones | 150 | Iteraciones del ciclo evolutivo |
| Tasa de Cruce | 0.80 | Probabilidad de recombinación BLX-alpha |
| Tasa de Mutación | 0.12 | Probabilidad de aplicar mutación creep |
| Élite (top N) | 8 | Individuos que pasan directos a la siguiente generación |
| Días históricos | 180 | Ventana temporal de datos de CoinGecko |
| Mín. por activo (%) | 5 | Asignación mínima por criptomoneda activa |
| Máx. por activo (%) | 60 | Asignación máxima por criptomoneda |
| Máx. en Short (%) | 40 | Exposición corta total máxima del portafolio |

---

## Fuente de Datos

| Componente | Detalle |
|------------|---------|
| **CoinGecko API v3** | Endpoint público, gratuito, sin clave. Precios de cierre diarios en USD. |
| **SQLite local** (`crypto_cache.db`) | Caché automática, válida por 24 h. Evita llamadas redundantes a la API. |
| **Modo simulado** | Datos sintéticos `N(μ, σ)` calibrados a rangos reales. Funciona offline. |

Criptomonedas disponibles: Bitcoin (BTC), Ethereum (ETH), Solana (SOL), BNB, XRP, Cardano (ADA), Avalanche (AVAX), Polkadot (DOT), Chainlink (LINK), Litecoin (LTC).

---

## Métricas del Portafolio Óptimo

| Métrica | Descripción |
|---------|-------------|
| **Sharpe Ratio** | Función objetivo del AG. Retorno por unidad de riesgo, anualizado. |
| **Retorno Anual Esperado** | `μₚ × 252 × 100%` sobre datos históricos. |
| **Volatilidad Anual** | `σₚ × √252 × 100%`. Riesgo total del portafolio. |
| **Máximo Drawdown** | Peor caída desde un pico acumulado en la ventana histórica. |
| **Retorno Diario Promedio** | Referencia operativa de corto plazo. |
| **N° posiciones LONG / SHORT** | Composición cualitativa del portafolio resultante. |

---

## Visualizaciones de la GUI

- **Convergencia del AG**: gráfico del Sharpe por generación (mejor y promedio) actualizado en tiempo real.
- **Portafolio Óptimo**: gráfico de pastel y barras horizontales. Las posiciones SHORT se muestran en rojo con barras hacia la izquierda.
- **Métricas**: tabla con colores por resultado (verde = positivo, rojo = negativo) y distribución en USD para una inversión de $1,000.

---

## Extensiones para Trabajo Final

```python
# Conectar a un exchange real con ccxt (Binance, Bybit, etc.)
import ccxt

exchange = ccxt.binance({'apiKey': 'KEY', 'secret': 'SECRET'})

def rebalance_portfolio(weights, names, capital_usdt=1000):
    for name, w in zip(names, weights):
        symbol = name.split('(')[1][:-1] + '/USDT'
        amount = abs(w) * capital_usdt
        if w > 0:
            exchange.create_market_buy_order(symbol, amount)
        else:
            exchange.create_market_sell_order(symbol, amount)  # short
```

Ideas de extensión:

1. **Backtesting**: simular el rendimiento del portafolio óptimo en períodos no vistos por el AG.
2. **Rebalanceo automático**: bot que re-ejecuta el AG cada semana y ajusta posiciones.
3. **Comparación de algoritmos**: AG vs Simulated Annealing vs optimización convexa (scipy).
4. **Multi-objetivo**: optimizar Sharpe y minimizar drawdown simultáneamente (NSGA-II).
5. **Análisis de correlaciones**: visualizar la matriz de correlación entre activos seleccionados.
6. **Stop-loss por posición**: cerrar automáticamente un short si el precio sube más de X%.

---

## Referencias

- Chang, T. J., Meade, N., Beasley, J. E., & Sharaiha, Y. M. (2000). Heuristics for cardinality constrained portfolio optimisation. *Computers & Operations Research, 27*(13), 1271–1302.
- Goldberg, D. E., & Deb, K. (1991). A comparative analysis of selection schemes used in genetic algorithms. *Foundations of Genetic Algorithms, 1*, 69–93.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77–91.
- Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business, 39*(1), 119–138.
- CoinGecko API v3: https://www.coingecko.com/en/api/documentation