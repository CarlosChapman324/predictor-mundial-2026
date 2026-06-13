# PLAN.md — Predictor Mundial 2026

Plan de implementación por fases. Objetivo: una app en Python que simula el Mundial 2026, proyecta los mercados de cada partido con un modelo propio, se valida contra torneos pasados y compara sus probabilidades con el mercado. Construir en orden: primero el núcleo riguroso (Capa 1), después los mercados extra (Capa 2).

---

## Arquitectura general

```
Ingesta de datos  ->  guarda en disco (Parquet / SQLite)
        |
   Modelo (matemática pura, sin red)  ->  predicciones con timestamp
        |
   Motor del torneo + Monte Carlo  ->  probabilidades de avance y campeón
        |
   Validación (backtesting)  y  Comparación vs mercado
        |
   Dashboard (Streamlit) lee de disco y renderiza
```

Tres responsabilidades separadas y desacopladas: ingesta (datos crudos), modelo (math puro, testeable sin internet), presentación (UI). El modelo nunca llama a una API; solo consume lo que ya está en disco.

Estructura de carpetas sugerida:
```
data/          ingesta y normalizacion
model/         modelo de goles, derivacion de mercados
tournament/    formato 2026, desempates, motor Monte Carlo
validation/    backtesting y metricas
market/        cuotas y probabilidades implicitas
app/           streamlit
scripts/       cron / recalculo
tests/         pytest
```

---

## Mercados

### Capa 1 — núcleo riguroso (construir primero)
Todos se derivan de la misma matriz de probabilidad de marcadores del modelo de goles:
- Resultado 1X2 (gana local / empate / gana visitante)
- Doble oportunidad (1X, 12, X2)
- Over/Under de goles (líneas 1.5, 2.5, 3.5)
- BTTS (ambos marcan): sí / no
- Clean sheet (gana a cero)
- Proyección de torneo: clasificación por grupo, avance por ronda y probabilidad de campeón (vía Monte Carlo)

### Capa 2 — mercados extra, experimentales (construir después, etiquetados como baja confianza)
Requieren submodelos aparte y datos granulares por partido y por jugador:
- Marcador exacto (top 5 más probables)
- Total de tarjetas (over/under, probabilidad de roja)
- Corners (over/under, línea dinámica)
- Goleadores (primer goleador y goleador en cualquier momento)
- Bota de Oro

---

## Fuentes de datos

### Capa 1 (gratis, sin API)
- Histórico de partidos internacionales: dataset público de resultados internacionales desde 1872 (formato CSV, p. ej. el de Kaggle de resultados internacionales). Incluye fecha, equipos, marcador, tipo de torneo y sede.
- Ratings de fuerza: calcular un Elo propio a partir del histórico (mejor para el portafolio que tomar uno prefabricado, y permite explicar el método). Como referencia/validación se puede contrastar con eloratings.net.
- Fixture oficial 2026: los 104 partidos, 12 grupos, sedes y el cuadro de eliminatorias. Tomar de la fuente oficial / Wikipedia y verificar contra el bracket publicado.

Nota: NO derivar la fuerza solo del ranking FIFA como hacen los ejemplos virales. El Elo histórico propio es la mejora.

### Capa 2 (API, posterior)
- API-Football (RapidAPI): lineups, eventos minuto a minuto, stats por partido (corners, tiros) y por jugador. Plan free: 100 requests/día, por eso el recálculo es diario y cacheado. Fallback: football-data.org.

---

## El modelo de goles (model/)

- **Base:** modelo de goles bivariado de Poisson con la corrección de **Dixon-Coles**. Es el estándar académico para fútbol y corrige la subestimación de empates y marcadores bajos que tiene el Poisson independiente. Esta corrección es uno de los diferenciadores frente a los ejemplos.
- **Parámetros a estimar:** una fuerza de ataque y una de defensa por selección, una ventaja de localía global (aplicada solo a los anfitriones USA, Canadá y México, ya que el resto juega en cancha neutral), y el parámetro de dependencia de marcadores bajos de Dixon-Coles.
- **Estimación:** máxima verosimilitud sobre el histórico, con **ponderación temporal** (los partidos recientes pesan más; usar un decaimiento exponencial tipo half-life). Ponderar también por relevancia del partido (un amistoso pesa menos que un partido oficial).
- **Goles esperados por partido:** a partir de las fuerzas ataque/defensa de los dos equipos más la localía se obtienen lambda_local y lambda_visitante.
- **Matriz de marcadores:** con esos dos lambda se construye la matriz de probabilidad de cada resultado i-j (con el ajuste de Dixon-Coles en los marcadores bajos). De esa única matriz se derivan TODOS los mercados de la Capa 1: 1X2 sumando las celdas correspondientes, over/under, BTTS, marcador exacto y clean sheet.

Tests con pytest: verificar que las probabilidades de cada mercado suman 1, que un equipo muy superior gana con probabilidad alta, que la localía mueve los números en la dirección correcta, etc.

---

## El motor del torneo (tournament/)

Esta es la parte más delicada. Implementar el formato 2026 con fidelidad y testearlo contra el cuadro oficial.

- **Formato:** 48 equipos, 12 grupos (A-L) de 4. Avanzan el primero y el segundo de cada grupo (24) más los 8 mejores terceros de los 12 (32 en total). Ronda de 32, octavos, cuartos, semifinales y final.
- **Desempates de grupo (orden 2026, verificar contra el reglamento oficial FIFA):** puntos, luego en caso de empate entre equipos los criterios de enfrentamiento directo (head-to-head) ANTES que la diferencia de goles global. Este cambio respecto a ediciones anteriores es importante: implementarlo mal rompe los grupos.
- **Mejores terceros:** rankearlos por los criterios globales y asignarlos a las llaves del cuadro según la tabla de combinaciones oficial de FIFA (qué grupos aportan terceros define a qué llave va cada uno).
- **Simulación de un torneo:** para cada partido, muestrear un resultado desde las probabilidades del modelo (o un marcador desde la matriz de Poisson), aplicar las reglas de avance y desempate, y resolver el cuadro hasta el campeón.
- **Monte Carlo:** correr el torneo completo al menos 10.000 veces. Agregar frecuencias para obtener probabilidad de clasificar por grupo, de llegar a cada ronda y de ser campeón.
- **Capa viva:** el Mundial ya arrancó (11 de junio de 2026). Los partidos ya jugados se fijan con su resultado real y solo se re-simulan los partidos pendientes. Las probabilidades se actualizan condicionadas a lo que ya pasó.

---

## Validación / backtesting (validation/) — DIFERENCIADOR

Probar que el modelo realmente funciona, no solo que produce números.

- Para cada torneo pasado de prueba (p. ej. Mundiales 2014, 2018, 2022 y Eurocopas recientes), entrenar el modelo SOLO con datos anteriores a ese torneo y predecir sus partidos.
- Métricas: **Ranked Probability Score (RPS)**, que es el más apropiado para el 1X2 por ser ordinal; además log-loss y Brier score. Reportar **calibración** con un reliability diagram (cuando el modelo dice 60%, ¿pasa el 60% de las veces?).
- Comparar contra baselines: predicción uniforme, fuerza por ranking FIFA y, si se consiguen, las cuotas del mercado. El objetivo es demostrar que el modelo le gana al azar y se acerca o supera a los baselines.

---

## Modelo vs mercado (market/) — DIFERENCIADOR

El ángulo financiero del proyecto.

- Obtener cuotas de casas de apuestas para los partidos (p. ej. The Odds API para las actuales; football-data.co.uk tiene cuotas históricas útiles para calibrar).
- Convertir las cuotas decimales a probabilidad implícita quitando el margen: p_i = (1/cuota_i) dividido por la suma de (1/cuota_j) de todos los resultados. Eso elimina el overround de la casa.
- Comparar la probabilidad del modelo con la implícita del mercado partido por partido. Mostrar dónde hay discrepancias (el modelo ve "valor").
- Análisis honesto y maduro: el mercado suele ser muy eficiente, así que las discrepancias grandes casi siempre indican un error o un dato faltante del modelo, no una oportunidad real. Enmarcarlo como estudio de eficiencia de mercado, no como sistema de apuestas.

---

## Fases

### Fase 0 — Setup (medio día)
Entorno Python, estructura de carpetas, dependencias, configuración de pytest. Esqueleto de los módulos vacíos.

### Fase 1 — Datos Capa 1 (1 día)
Descargar y normalizar el histórico de partidos. Calcular el Elo propio. Cargar el fixture oficial 2026 (grupos, sedes, cuadro). Guardar todo en disco.

### Fase 2 — Modelo de goles (1-2 días)
Implementar el modelo de Poisson con Dixon-Coles y la estimación por máxima verosimilitud con ponderación temporal. Derivar de la matriz de marcadores todos los mercados de la Capa 1. Tests.

### Fase 3 — Motor del torneo + Monte Carlo (1-2 días)
Implementar el formato 2026 con sus desempates y la asignación de mejores terceros. Loop de simulación y Monte Carlo de 10.000 corridas. Testear el formato contra el cuadro oficial.

### Fase 4 — Validación / backtesting (1 día)
Backtesting contra torneos pasados con RPS, log-loss y calibración frente a baselines.

### Fase 5 — Modelo vs mercado (medio día)
Ingesta de cuotas, conversión a probabilidad implícita y comparación con el modelo.

### Fase 6 — Dashboard Streamlit, Capa 1 (2 días)
Pestañas: campeón y título odds, fase de grupos (clasificación proyectada), cuadro de eliminatorias, vista por partido con los mercados de la Capa 1, y dos pestañas propias de este proyecto: **Validación** (las métricas del backtesting) y **Mercado** (modelo vs cuotas). Diseño oscuro tipo terminal deportiva, números grandes, visualizaciones custom, acento azul.

### Fase 7 — Capa viva (medio día)
Fijar resultados ya jugados y re-simular solo lo pendiente. Guardar predicciones con timestamp para mostrar el histórico.

### Fase 8 — Capa 2, mercados extra (opcional, posterior)
Conectar API-Football. Submodelos de tarjetas y corners (regresión sobre promedios ajustada por intensidad) y de goleadores (repartir el lambda del equipo entre jugadores según sus minutos y su tasa de gol). Mostrarlos SIEMPRE etiquetados como experimentales y con confianza baja.

### Fase 9 — Pulido y deploy (medio día)
Animaciones suaves, banner "Análisis estadístico de entretenimiento. No es asesoría de apuestas.", modo captura limpio para los clips, README claro en GitHub que explique el modelo y la validación. Deploy a Streamlit Community Cloud.

---

## Riesgos

- **Validación necesita datos de torneos pasados:** asegurar que el histórico cubre los torneos de prueba y que el corte temporal del entrenamiento es estricto (no usar datos del futuro).
- **Cuotas históricas:** pueden ser difíciles de conseguir para todos los partidos; si faltan, limitar la comparación de mercado a los partidos con cuotas disponibles y dejarlo claro.
- **Calidad por mercado:** Poisson es sólido para goles y resultado, más ruidoso para tarjetas, corners y goleadores. Esos mercados van en la Capa 2 y se muestran con confianza baja.
- **Límite de la API en Capa 2:** el recálculo diario con cache respeta el plan free; datos en vivo durante los partidos requerirían plan pago.
- **Legal:** mientras la app solo muestre probabilidades y no reciba dinero, es análisis estadístico de entretenimiento.
