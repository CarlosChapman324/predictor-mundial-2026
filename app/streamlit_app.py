"""Dashboard de Streamlit (Fase 6).

Estetica: terminal de datos deportiva (estilo broadcast / Opta), fondo oscuro
con acento azul, numeros grandes como protagonistas, visualizaciones propias.
Pestanas: campeon, fase de grupos, cuadro, vista por partido, Validacion y
Mercado. Solo presentacion: lee los Parquet de disco, no calcula el modelo.

Pendiente de implementar en la Fase 6. Por ahora es un marcador de posicion.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st


def main():
    st.set_page_config(page_title="Predictor Mundial 2026", layout="wide")
    st.title("Predictor Mundial 2026")
    st.caption("Analisis estadistico de entretenimiento. No es asesoria de apuestas.")
    st.info("Dashboard en construccion. Se implementa en la Fase 6.")


if __name__ == "__main__":
    main()
