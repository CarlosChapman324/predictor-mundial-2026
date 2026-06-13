# Predictor Mundial 2026

Simulacion y proyeccion estadistica de la Copa Mundial de la FIFA 2026 con un
modelo propio. Cubre los 104 partidos, la fase de grupos, el cuadro de
eliminatorias y la probabilidad de campeon, y compara las probabilidades del
modelo con las del mercado de apuestas.

> Analisis estadistico de entretenimiento. No es una casa de apuestas ni
> asesoria de apuestas: solo muestra probabilidades.

## Que lo hace distinto

A diferencia de los predictores virales que circulan, este proyecto prioriza el
**rigor** sobre la cantidad de mercados llamativos:

1. **Validacion / backtesting.** El modelo se prueba contra Mundiales y
   Eurocopas pasados y se reporta su desempeno con metricas formales (Ranked
   Probability Score, log-loss, calibracion) frente a baselines.
2. **Modelo vs mercado.** Las probabilidades del modelo se comparan con las
   probabilidades implicitas de las cuotas (quitando el margen), como analisis
   de eficiencia de mercado.
3. **Honestidad estadistica.** Cada mercado muestra su nivel de confianza; los
   ruidosos se etiquetan como experimentales.

## El modelo

Goles bivariados de Poisson con la correccion de **Dixon-Coles**, estimado por
maxima verosimilitud sobre el historico internacional con ponderacion temporal.
La fuerza de cada seleccion se mide con un **Elo propio** calculado desde el
registro de partidos (no se toma un ranking prefabricado).

## Stack

Python 3.11+, pandas/numpy, scipy/statsmodels, Streamlit, Plotly. Entorno y
dependencias gestionados con [uv](https://github.com/astral-sh/uv).

## Estructura

```
data/          ingesta y normalizacion de datos (historico, fixture, Elo)
model/          modelo de goles y derivacion de mercados (matematica pura)
tournament/     formato 2026, desempates y motor Monte Carlo
validation/     backtesting y metricas
market/         cuotas y probabilidades implicitas
app/            dashboard de Streamlit
scripts/        orquestacion (descarga, recalculo)
tests/          pytest
```

## Como correrlo

```bash
# Instalar el entorno (crea .venv con Python 3.12 e instala dependencias)
uv sync

# Construir los datos de la Capa 1 (historico + Elo + fixture)
uv run python -m scripts.build_data

# Correr los tests
uv run pytest
```

## Estado

En construccion por fases. El detalle de cada fase vive en `PLAN.md`.

- [x] Fase 0 — Setup del entorno y estructura
- [x] Fase 1 — Datos Capa 1 (historico, Elo, fixture)
- [ ] Fase 2 — Modelo de goles (Poisson + Dixon-Coles)
- [ ] Fase 3 — Motor del torneo + Monte Carlo
- [ ] Fase 4 — Validacion / backtesting
- [ ] Fase 5 — Modelo vs mercado
- [ ] Fase 6 — Dashboard Streamlit
- [ ] Fase 7 — Capa viva
