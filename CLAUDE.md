# CLAUDE.md — Predictor Mundial 2026 (proyecto de portafolio)

## Qué es
Aplicación que simula el Mundial 2026 y proyecta los mercados de cada partido con un modelo estadístico propio. Cubre los 104 partidos, las 12 fases de grupo, el cuadro completo de eliminatorias y la probabilidad de campeón, y actualiza las probabilidades a medida que se juegan los partidos reales.

A diferencia de los predictores "virales" que circulan, este proyecto es una **pieza de portafolio profesional** para aplicar a roles de finanzas y análisis de datos en Europa. El énfasis NO está en mostrar la mayor cantidad de mercados llamativos, sino en el **rigor**: un modelo bien fundamentado, validado contra torneos pasados y honesto sobre su incertidumbre.

Framing público: análisis estadístico de entretenimiento. NO es una casa de apuestas, no recibe ni procesa dinero, solo muestra probabilidades.

## Diferenciadores (esto es lo que da valor al proyecto, no lo descuides)
1. **Validación / backtesting.** El modelo se prueba contra torneos pasados (Mundiales y Eurocopas anteriores) y se reporta su desempeño con métricas formales frente a baselines. Ningún clon hace esto y es lo que más pesa en una entrevista técnica.
2. **Modelo vs mercado.** Las probabilidades del modelo se comparan con las probabilidades implícitas de las cuotas de las casas de apuestas (quitando el margen), como análisis de eficiencia de mercado. Es el ángulo que conecta el proyecto con finanzas.
3. **Honestidad estadística.** Cada mercado muestra su nivel de confianza; los mercados ruidosos se etiquetan como experimentales y nunca se presentan con la misma autoridad que los sólidos.

## Stack
- Python 3.11+
- pandas, numpy para manejo de datos
- scipy y statsmodels para ajustar el modelo por máxima verosimilitud
- Streamlit para el dashboard interactivo
- Plotly o Altair para visualizaciones propias (NO usar gráficos por defecto con look genérico)
- requests solo para la ingesta de datos de la Capa 2
- pytest para tests del modelo
- Almacenamiento local en Parquet o SQLite para cachear datos y predicciones

## Convenciones
- El modelo vive aislado en `model/` y `tournament/`: matemática pura, sin llamadas de red. Debe poder testearse sin internet.
- La ingesta de datos vive en `data/`, la validación en `validation/`, la app de Streamlit en `app/`.
- Toda predicción se guarda con timestamp para poder mostrar cómo se movieron las probabilidades a lo largo del torneo.
- Código legible y comentado: el dev tiene que poder explicar cada decisión en una entrevista. Prioriza claridad sobre astucia.
- Nada de "--" (doble guion) en código ni en texto.
- Funciones puras y testeables; la lógica del modelo nunca se mezcla con la presentación.

## Diseño
La UI no puede verse como el típico dashboard genérico de IA. Norte estético: **terminal de datos deportiva** (estilo broadcast / Opta), fondo oscuro con un acento, números grandes como protagonistas, visualizaciones propias y limpias. Mantén UNA sola dirección visual coherente en todas las pantallas. Usa un acento azul para alinear con el CV del dev.

## Contexto del dev
Carlos Chapman, economista (Barranquilla, Colombia), fuerte en R y aprendiendo Python para diversificar su perfil. Este es el proyecto estrella de su portafolio para roles de finanzas y datos en Europa. Quiere ENTENDER lo que se construye, no solo ejecutarlo: explica el porqué de las decisiones técnicas a medida que avanzas, y construye por fases para que pueda seguir cada paso.

El detalle de implementación por fases vive en PLAN.md.
