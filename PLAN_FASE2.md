# PLAN_FASE2.md — Capa 2 (mercados extra) + Modulo de valor

Continuacion del PLAN.md. La Capa 1 ya esta construida y desplegada (1X2, goles,
BTTS, gana a cero, campeon, validacion y comparacion de mercado en 1X2). Esta
segunda vuelta anade tres cosas: los mercados de props como prediccion del
modelo, el factor arbitro para tarjetas, y un modulo de analisis de valor sobre
los mercados que si tienen cuotas.

Norte: pieza de portafolio. Todo con framing de analisis de valor honesto y la
advertencia de eficiencia de mercado. No es un tablero de pronosticos de apuestas.

## Decisiones ya tomadas (no reabrir)
- Value bet (comparacion contra el mercado) SOLO en 1X2, doble oportunidad y
  totales de goles (mercados con cuotas gratis en The Odds API). NO pagar por
  cuotas de props.
- Corners, tarjetas, remates y remates a puerta se calculan como prediccion del
  modelo, sin value bet, y se muestran SIEMPRE etiquetados experimentales y de
  confianza baja.
- Tarjetas con el historial individual de cada arbitro (de API-Football), no por
  categoria continental. El arbitro es la variable de mayor peso del submodelo.
- Toda la seccion de valor lleva la nota de que el mercado es eficiente y una
  discrepancia grande suele delatar un dato que le falta al modelo, no una
  oportunidad real.

## Fuentes de datos nuevas
1. API-Football (RapidAPI): stats historicas por partido y equipo (corners,
   remates, remates a puerta, tarjetas, faltas), designaciones de arbitros del
   Mundial 2026 e historial de tarjetas por arbitro. Cachear en disco. Respetar
   el limite free (100 requests/dia): traer solo lo necesario, cachear agresivo.
2. Cuotas: The Odds API free (1X2 y totales de goles). Para el backtest historico
   de la estrategia de valor, cuotas historicas de football-data.co.uk.

## Submodelos de Capa 2 (model/props/)
Modelos de conteo; toda la salida marcada confianza baja.
- Corners: Poisson sobre las tasas historicas de corners a favor y en contra de
  cada equipo, ajustada por la fuerza relativa de los dos equipos. Over/under.
- Remates y remates a puerta: mismo enfoque, Poisson sobre tasas de tiros por
  equipo, ajustado por fuerza relativa.
- Tarjetas: conteo con dos componentes: (a) propension de los equipos (faltas y
  tarjetas historicas) y (b) factor arbitro (promedio de tarjetas por partido del
  arbitro designado). El arbitro pesa mas. Over/under de tarjetas y probabilidad
  de roja.
Tests: cantidades esperadas en rangos sensatos y el factor arbitro mueve las
tarjetas en la direccion correcta (un arbitro de gatillo facil sube la expectativa).

## Modulo de valor (value/) — el diferenciador de esta fase
- EV = p_modelo * cuota - 1 para cada seleccion de 1X2, doble oportunidad y
  totales de goles. Marcar las de EV positivo.
- Mejor apuesta del partido: la de mayor EV entre los mercados con cuotas, con su
  EV y un nivel de confianza derivado de la calibracion del modelo en ese mercado.
  Sin props (no tienen cuotas, no se puede medir su valor).
- Backtest de la estrategia de valor sobre torneos pasados (con cuotas historicas):
  ROI, yield y evolucion del bankroll, contra dos baselines (apostar siempre al
  favorito y apostar al azar). Reportar con franqueza: lo mas probable es edge
  marginal o negativo tras el margen, y decirlo es parte del valor del proyecto.
- Opcional: criterio de Kelly fraccionado para dimensionar la apuesta.

## Integracion en la UI (app/)
- Vista "Por partido": seccion de Capa 2 (corners, tarjetas, remates, remates a
  puerta) como prediccion del modelo, con etiqueta visible de "experimental". Si
  el partido ya tiene arbitro designado, mostrarlo junto a su promedio de tarjetas.
- Indicador de valor por partido: en los mercados con cuotas, resaltar donde el
  modelo ve valor (EV positivo), con confianza y la advertencia de eficiencia.
- Pestana nueva "Valor" (opcional): mayores discrepancias modelo vs mercado y los
  resultados del backtest de la estrategia (ROI, bankroll, baselines).
Misma direccion visual (terminal de datos deportiva, fondo oscuro, acento azul).

## Automatizacion (cron)
GitHub Actions con schedule diario que corre el pipeline completo (resultados,
cuotas y stats; recalcula modelo, simulacion y valor; guarda predicciones con
timestamp) y commitea los datos actualizados al repo. Streamlit Cloud redespliega
solo. Claves (THE_ODDS_API_KEY, RAPIDAPI_KEY) en los secrets de GitHub Actions,
nunca en el codigo. El .env permanece en .gitignore.

## Fases
- Fase 1 — Datos de Capa 2 (1 dia): cliente de API-Football con retry y cache;
  ingesta de stats historicas por partido, designaciones de arbitros del Mundial
  e historial de tarjetas por arbitro. Guardar en disco.
- Fase 2 — Submodelos de props (1 a 2 dias): corners, remates y remates a puerta
  (Poisson sobre tasas ajustadas por fuerza); tarjetas con el factor arbitro.
  Tests. Salida etiquetada confianza baja.
- Fase 3 — Modulo de valor (1 dia): EV en los mercados con cuotas, "mejor apuesta
  del partido" con su confianza, y backtest de la estrategia con ROI vs baselines.
  Kelly opcional.
- Fase 4 — Integracion UI (1 a 2 dias): props en la vista por partido (con el
  arbitro cuando exista), indicador de valor y, opcional, la pestana "Valor".
- Fase 5 — Automatizacion y deploy (medio dia): GitHub Actions con cron diario,
  claves en secrets y re-deploy.

## Riesgos
- Limite de la API free: las stats historicas consumen muchas requests; cachear
  agresivo y traer solo lo indispensable. El cron diario respeta el plan free.
- Designaciones de arbitros: pueden no estar publicadas con antelacion; fallback
  al promedio general de tarjetas hasta conocer el arbitro, indicado en la UI.
- Props ruidosos: siempre confianza baja, nunca con la autoridad del resultado.
- Honestidad del value bet: el backtest probablemente mostrara edge marginal o
  negativo tras el margen; reportarlo con franqueza es parte del valor.

## Convenciones
Las mismas del CLAUDE.md: el modelo y los submodelos viven aislados y sin llamadas
de red (testeables sin internet), todo con tests, nada de doble guion en codigo ni
texto, codigo legible y comentado para poder defenderlo en una entrevista.
