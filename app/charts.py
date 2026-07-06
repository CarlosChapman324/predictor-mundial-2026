"""Visualizaciones propias en Plotly, con la estetica del dashboard.

Nada de graficos por defecto: cada figura usa la paleta y el estilo comun
(fondo transparente, acento azul, rejilla tenue) definidos en app/theme.py.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app.theme import BLUE_SCALE, CATEGORICAL, COLORS, style_fig

# Rondas del torneo en orden, con su etiqueta para el dashboard.
ROUND_COLUMNS = [
    ("qualify", "Clasifica"),
    ("round_of_16", "Octavos"),
    ("quarterfinal", "Cuartos"),
    ("semifinal", "Semis"),
    ("final", "Final"),
    ("champion", "Campeon"),
]


def champion_bar(sim, n: int = 15):
    """Barras horizontales de probabilidad de campeon (top n)."""
    d = sim.nlargest(n, "champion").iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["champion"] * 100, y=d["team"], orientation="h",
        marker=dict(color=d["champion"], colorscale=BLUE_SCALE, line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in d["champion"]],
        textposition="outside", textfont=dict(color=COLORS["text"]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Probabilidad de ser campeon")
    fig.update_xaxes(ticksuffix="%", showgrid=True)
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=470)


def advance_heatmap(sim, n: int = 16):
    """Mapa de calor: probabilidad de cada equipo de alcanzar cada ronda."""
    d = sim.nlargest(n, "champion")
    cols = [c for c, _ in ROUND_COLUMNS]
    z = d[cols].to_numpy() * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=[lbl for _, lbl in ROUND_COLUMNS], y=d["team"],
        colorscale=BLUE_SCALE, zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=11), hovertemplate="%{y} &middot; %{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(ticksuffix="%", outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(title="Camino al titulo: probabilidad de alcanzar cada ronda")
    return style_fig(fig, height=540)


def group_bars(group_df):
    """Barras agrupadas de P(clasifica) y P(gana grupo) de los 4 equipos."""
    d = group_df.sort_values("qualify")
    fig = go.Figure()
    fig.add_bar(y=d["team"], x=d["qualify"] * 100, orientation="h",
                name="Clasifica", marker_color=COLORS["accent"])
    fig.add_bar(y=d["team"], x=d["win_group"] * 100, orientation="h",
                name="Gana grupo", marker_color=COLORS["away"])
    fig.update_layout(barmode="group", title="Clasificacion proyectada",
                      legend=dict(orientation="h", y=1.15, x=0))
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=300)


def one_x_two_bar(p_home, p_draw, p_away, home, away):
    """Barra apilada del resultado 1X2 de un partido."""
    fig = go.Figure()
    for value, label, color in [
        (p_home, home, COLORS["home"]),
        (p_draw, "Empate", COLORS["draw"]),
        (p_away, away, COLORS["away"]),
    ]:
        fig.add_bar(x=[value * 100], y=["1X2"], orientation="h", name=label,
                    marker_color=color, text=f"{value * 100:.0f}%", textposition="inside",
                    insidetextanchor="middle", hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>")
    fig.update_layout(barmode="stack", title="Resultado 1X2",
                      legend=dict(orientation="h", y=-0.25, x=0))
    fig.update_xaxes(visible=False, range=[0, 100])
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=170)


def secondary_markets_bar(row):
    """Barras horizontales de los mercados secundarios de un partido."""
    items = [
        ("Mas de 1.5 goles", row["over_1_5"]),
        ("Mas de 2.5 goles", row["over_2_5"]),
        ("Mas de 3.5 goles", row["over_3_5"]),
        ("Ambos marcan", row["btts_yes"]),
        (f"Cero {row['home_team']}", row["cs_home"]),
        (f"Cero {row['away_team']}", row["cs_away"]),
    ]
    labels = [k for k, _ in items][::-1]
    values = [v * 100 for _, v in items][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale=BLUE_SCALE, cmin=0, cmax=100),
        text=[f"{v:.0f}%" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Otros mercados")
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=300)


def calibration_diagram(cal):
    """Reliability diagram: probabilidad predicha vs frecuencia observada."""
    sizes = 8 + 26 * (cal["count"] / cal["count"].max()) ** 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=COLORS["muted"], dash="dash"),
                             name="Calibracion perfecta", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=cal["mean_predicted"], y=cal["observed_frequency"], mode="markers+lines",
        line=dict(color=COLORS["accent"]),
        marker=dict(size=sizes, color=COLORS["accent"], line=dict(color=COLORS["bg"], width=1)),
        name="Modelo", hovertemplate="Predicha %{x:.0%} &middot; observada %{y:.0%}<extra></extra>",
    ))
    fig.update_layout(title="Calibracion del modelo", showlegend=True,
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(title="Probabilidad predicha", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Frecuencia observada", tickformat=".0%", range=[0, 1])
    return style_fig(fig, height=430)


def rps_bars(summary):
    """RPS por torneo, comparando modelo vs baselines (menor es mejor)."""
    d = summary[summary["tournament"] != "Todos"]
    series = [("model", "Modelo", COLORS["accent"]),
              ("elo", "Elo", COLORS["away"]),
              ("uniform", "Uniforme", COLORS["draw"])]
    fig = go.Figure()
    for key, label, color in series:
        sub = d[d["predictor"] == key]
        fig.add_bar(x=sub["tournament"], y=sub["rps"], name=label, marker_color=color)
    fig.update_layout(barmode="group", title="RPS por torneo (menor es mejor)",
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(title="RPS")
    fig.update_xaxes(showgrid=False)
    return style_fig(fig, height=430)


def market_scatter(comp):
    """Dispersion modelo vs mercado: cada punto es un resultado de un partido."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=COLORS["muted"], dash="dash"),
                             name="Acuerdo total", hoverinfo="skip"))
    label = comp["home_team"] + " vs " + comp["away_team"]
    for outcome, color, name in [("home", COLORS["home"], "Local"),
                                 ("draw", COLORS["draw"], "Empate"),
                                 ("away", COLORS["away"], "Visitante")]:
        fig.add_trace(go.Scatter(
            x=comp[f"market_{outcome}"], y=comp[f"model_{outcome}"], mode="markers",
            name=name, marker=dict(color=color, size=8, opacity=0.7, line=dict(width=0)),
            text=label, hovertemplate="%{text}<br>Mercado %{x:.0%} &middot; modelo %{y:.0%}<extra></extra>",
        ))
    fig.update_layout(title="Modelo vs mercado por resultado",
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(title="Probabilidad del mercado", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Probabilidad del modelo", tickformat=".0%", range=[0, 1])
    return style_fig(fig, height=470)


def probability_evolution(history, n: int = 8):
    """Lineas de la probabilidad de campeon a medida que avanza el torneo.

    Eje x: partidos jugados (0 = pronostico sin condicionar). Una linea por cada
    una de las n selecciones mas probables en la foto mas reciente.
    """
    latest = history["matches_played"].max()
    top = history[history["matches_played"] == latest].nlargest(n, "champion")["team"].tolist()
    fig = go.Figure()
    for i, team in enumerate(top):
        d = history[history["team"] == team].sort_values("matches_played")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        fig.add_trace(go.Scatter(
            x=d["matches_played"], y=d["champion"] * 100, mode="lines+markers", name=team,
            line=dict(color=color, width=2), marker=dict(color=color, size=7),
            hovertemplate=f"{team}<br>%{{x}} jugados &middot; %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(title="Evolucion de la probabilidad de campeon")
    fig.update_xaxes(title="Partidos jugados (grupos + eliminatoria)", showgrid=False)
    fig.update_yaxes(title="Campeon", ticksuffix="%")
    style_fig(fig, height=470)
    fig.update_layout(colorway=CATEGORICAL)
    return fig


def golden_boot_bar(golden, n: int = 12):
    """Barras horizontales de probabilidad de Bota de Oro (top n goleadores)."""
    d = golden.head(n).iloc[::-1]
    labels = [f"{r.player} ({r.team})" for r in d.itertuples(index=False)]
    fig = go.Figure(go.Bar(
        x=d["win_prob"] * 100, y=labels, orientation="h",
        marker=dict(color=d["win_prob"], colorscale=BLUE_SCALE, line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in d["win_prob"]],
        textposition="outside", textfont=dict(color=COLORS["text"]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Probabilidad de Bota de Oro (experimental)")
    fig.update_xaxes(ticksuffix="%", showgrid=True)
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=440)
