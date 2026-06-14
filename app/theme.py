"""Estetica del dashboard: terminal de datos deportiva (estilo broadcast / Opta).

Fondo oscuro, un unico acento azul (alineado con el CV), numeros grandes como
protagonistas y una sola direccion visual coherente. Aqui viven la paleta, el
CSS global y un ayudante para dar el mismo estilo a todas las figuras de Plotly.
"""

from __future__ import annotations

import streamlit as st

COLORS = {
    "bg": "#0A0E16",
    "panel": "#121A2A",
    "border": "#21304A",
    "text": "#E6EDF3",
    "muted": "#8295B0",
    "grid": "#1E2B42",
    "accent": "#2F81F7",      # azul principal
    "accent_soft": "#5BA8FF",
    "home": "#2F81F7",        # gana local
    "draw": "#6E7E99",        # empate
    "away": "#E8A23D",        # gana visitante (ambar, para contraste)
    "good": "#3FB950",
    "bad": "#F85149",
}

# Escala azul para barras y mapas de calor.
BLUE_SCALE = [[0.0, "#0E1B30"], [0.35, "#16447E"], [0.7, "#2F81F7"], [1.0, "#7CC0FF"]]

FONT = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Menlo, Consolas, monospace"


def inject_css() -> None:
    """Inyecta el CSS global. Da el look de terminal y limpia el chrome de Streamlit."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=IBM+Plex+Mono:wght@500;700&display=swap');

        .stApp {{ background: {COLORS['bg']}; color: {COLORS['text']}; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }}

        html, body, [class*="css"] {{ font-family: {FONT}; }}

        /* Encabezado del dashboard */
        .pm-title {{ font-family: {MONO}; font-weight: 700; font-size: 2.0rem;
                     letter-spacing: 0.04em; color: {COLORS['text']}; margin: 0; }}
        .pm-title .accent {{ color: {COLORS['accent']}; }}
        .pm-sub {{ color: {COLORS['muted']}; font-size: 0.9rem; margin-top: 0.15rem; }}
        .pm-banner {{ display:inline-block; margin-top:0.6rem; padding:0.25rem 0.7rem;
                      border:1px solid {COLORS['border']}; border-radius:999px;
                      color:{COLORS['muted']}; font-size:0.72rem; letter-spacing:0.03em; }}

        /* Tarjetas KPI */
        .pm-kpi {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']};
                   border-radius: 12px; padding: 0.9rem 1.1rem; }}
        .pm-kpi .label {{ color: {COLORS['muted']}; font-size: 0.72rem; text-transform: uppercase;
                          letter-spacing: 0.08em; }}
        .pm-kpi .value {{ font-family: {MONO}; font-weight: 700; font-size: 2.1rem;
                          color: {COLORS['text']}; line-height: 1.1; }}
        .pm-kpi .value .accent {{ color: {COLORS['accent']}; }}
        .pm-kpi .sub {{ color: {COLORS['muted']}; font-size: 0.78rem; }}

        /* Pestanas */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; border-bottom: 1px solid {COLORS['border']}; }}
        .stTabs [data-baseweb="tab"] {{ color: {COLORS['muted']}; font-size: 0.86rem;
                                        padding: 0.5rem 0.9rem; }}
        .stTabs [aria-selected="true"] {{ color: {COLORS['text']};
                                          border-bottom: 2px solid {COLORS['accent']}; }}

        /* Tablas */
        [data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: 10px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    """Encabezado fijo con titulo, subtitulo y el aviso legal."""
    st.markdown(
        f"""
        <div class="pm-title">PREDICTOR <span class="accent">MUNDIAL</span> 2026</div>
        <div class="pm-sub">Modelo estadistico propio &middot; Poisson + Dixon-Coles &middot;
        validado contra torneos pasados y contra el mercado</div>
        <div class="pm-banner">Analisis estadistico de entretenimiento. No es asesoria de apuestas.</div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    """Devuelve el HTML de una tarjeta KPI (numero grande protagonista)."""
    value_html = f'<span class="accent">{value}</span>' if accent else value
    return (
        f'<div class="pm-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value_html}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def style_fig(fig, height: int | None = None):
    """Aplica la estetica comun a una figura de Plotly (fondo transparente, rejilla tenue)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family=FONT, size=13),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
        colorway=[COLORS["accent"], COLORS["away"], COLORS["draw"], COLORS["good"]],
        hoverlabel=dict(bgcolor=COLORS["panel"], font=dict(color=COLORS["text"], family=FONT)),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["border"])
    if height:
        fig.update_layout(height=height)
    return fig
