"""Dashboard del Predictor Mundial 2026 (Streamlit).

Estetica de terminal de datos deportiva: fondo oscuro, acento azul, numeros
grandes y visualizaciones propias. Solo PRESENTACION: lee los Parquet de disco
que generan los scripts de las fases previas y los renderiza. No ejecuta el
modelo ni toca la red.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permite "from app import ..." al ejecutarse como script de Streamlit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from app import charts, data, theme  # noqa: E402

st.set_page_config(page_title="Predictor Mundial 2026", layout="wide", page_icon="*")
theme.inject_css()
theme.header()

sim = data.simulation()
if sim is None:
    st.error("Faltan datos. Corre primero:  uv run python -m scripts.build_simulation")
    st.stop()

meta = data.simulation_meta() or {}
st.caption(
    f"{meta.get('n_sims', 0):,} simulaciones Monte Carlo &middot; "
    f"{meta.get('fixture_played', 0)} partidos ya jugados fijados"
)


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


tab_champ, tab_groups, tab_path, tab_match, tab_val, tab_market = st.tabs(
    ["Campeon", "Grupos", "Camino al titulo", "Por partido", "Validacion", "Mercado"]
)


# --- Campeon ---------------------------------------------------------------
with tab_champ:
    top3 = sim.nlargest(3, "champion").reset_index(drop=True)
    cols = st.columns(3)
    for col, (_, row) in zip(cols, top3.iterrows()):
        col.markdown(
            theme.kpi(row["team"], pct(row["champion"]), f"Final {pct(row['final'])}", accent=True),
            unsafe_allow_html=True,
        )
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.champion_bar(sim, n=15), width="stretch")
    with right:
        table = sim.head(20)[["team", "champion", "final", "semifinal", "qualify"]].copy()
        table.columns = ["Seleccion", "Campeon", "Final", "Semis", "Clasifica"]
        st.dataframe(
            table.style.format({c: "{:.1%}" for c in ["Campeon", "Final", "Semis", "Clasifica"]}),
            width="stretch", hide_index=True, height=470,
        )


# --- Grupos ----------------------------------------------------------------
with tab_groups:
    groups = data.groups()
    if not groups:
        st.info("Falta groups.json.")
    else:
        choice = st.selectbox("Grupo", list(groups.keys()), format_func=lambda g: f"Grupo {g}")
        members = groups[choice]
        gdf = sim[sim["team"].isin(members)].copy()
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.group_bars(gdf), width="stretch")
        with right:
            t = gdf.sort_values("qualify", ascending=False)[
                ["team", "win_group", "runner_up", "qualify"]
            ].copy()
            t.columns = ["Seleccion", "Gana grupo", "2o", "Clasifica"]
            st.dataframe(
                t.style.format({c: "{:.1%}" for c in ["Gana grupo", "2o", "Clasifica"]}),
                width="stretch", hide_index=True, height=200,
            )
        st.caption("Clasifican el 1o y el 2o de cada grupo, mas los 8 mejores terceros de los 12.")


# --- Camino al titulo ------------------------------------------------------
with tab_path:
    st.plotly_chart(charts.advance_heatmap(sim, n=16), width="stretch")
    st.caption("Probabilidad de cada seleccion de superar cada ronda, agregada sobre las simulaciones.")


# --- Por partido -----------------------------------------------------------
with tab_match:
    mm = data.match_markets()
    if mm is None:
        st.info("Faltan los mercados por partido. Corre: uv run python -m scripts.build_simulation")
    else:
        mm = mm.sort_values("date")
        labels = [
            f"{r.date.strftime('%d %b')} &middot; {r.home_team} vs {r.away_team}".replace("&middot;", "|")
            for r in mm.itertuples(index=False)
        ]
        idx = st.selectbox("Partido", range(len(mm)), format_func=lambda i: labels[i])
        row = mm.iloc[idx]

        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi(row["home_team"], pct(row["p_home"], 0), "gana", accent=True), unsafe_allow_html=True)
        c2.markdown(theme.kpi("Empate", pct(row["p_draw"], 0)), unsafe_allow_html=True)
        c3.markdown(theme.kpi(row["away_team"], pct(row["p_away"], 0), "gana", accent=True), unsafe_allow_html=True)

        sub = f"Goles esperados: {row['lambda_home']:.2f} - {row['lambda_away']:.2f}"
        if bool(row["played"]):
            sub += f"  |  Resultado real: {int(row['home_score'])}-{int(row['away_score'])}"
        st.caption(sub)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.one_x_two_bar(row["p_home"], row["p_draw"], row["p_away"], row["home_team"], row["away_team"]),
                width="stretch",
            )
            scores = json.loads(row["exact_scores"])
            lines = " &nbsp; ".join(f"{s['home_goals']}-{s['away_goals']} ({pct(s['prob'], 0)})" for s in scores)
            st.markdown(f"**Marcadores mas probables:** {lines}", unsafe_allow_html=True)
        with right:
            st.plotly_chart(charts.secondary_markets_bar(row), width="stretch")


# --- Validacion ------------------------------------------------------------
with tab_val:
    summary = data.backtest_summary()
    calibration = data.backtest_calibration()
    if summary is None:
        st.info("Falta el backtesting. Corre: uv run python -m scripts.build_validation")
    else:
        overall = summary[summary["tournament"] == "Todos"].set_index("predictor")
        model_rps = overall.loc["model", "rps"]
        skill_unif = 1 - model_rps / overall.loc["uniform", "rps"]
        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi("RPS del modelo", f"{model_rps:.3f}", "menor es mejor", accent=True), unsafe_allow_html=True)
        c2.markdown(theme.kpi("Skill vs azar", pct(skill_unif), "mejora sobre el uniforme"), unsafe_allow_html=True)
        c3.markdown(theme.kpi("RPS del mercado Elo", f"{overall.loc['elo', 'rps']:.3f}", "baseline de fuerza"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.rps_bars(summary), width="stretch")
        with right:
            if calibration is not None and not calibration.empty:
                st.plotly_chart(charts.calibration_diagram(calibration), width="stretch")
        st.caption("Entrenando solo con datos anteriores a cada torneo (corte temporal estricto).")


# --- Mercado ---------------------------------------------------------------
with tab_market:
    comp = data.market_comparison()
    if comp is None or comp.empty:
        st.info(
            "Aun no hay comparacion de mercado. Consigue cuotas (THE_ODDS_API_KEY o un CSV) "
            "y corre: uv run python -m scripts.build_market"
        )
    else:
        model = comp[["model_home", "model_draw", "model_away"]].to_numpy().ravel()
        market = comp[["market_home", "market_draw", "market_away"]].to_numpy().ravel()
        corr = float(np.corrcoef(model, market)[0, 1])
        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi("Margen de las casas", pct(comp["overround"].mean()), "overround medio"), unsafe_allow_html=True)
        c2.markdown(theme.kpi("Acuerdo en el favorito", pct(comp["same_favorite"].mean(), 0), "modelo vs mercado", accent=True), unsafe_allow_html=True)
        c3.markdown(theme.kpi("Correlacion", f"{corr:.3f}", "modelo vs mercado"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.market_scatter(comp), width="stretch")
        with right:
            top = comp.nlargest(10, "max_abs_edge")[["home_team", "away_team", "model_home", "market_home"]].copy()
            top.columns = ["Local", "Visitante", "Modelo (L)", "Mercado (L)"]
            st.dataframe(
                top.style.format({"Modelo (L)": "{:.0%}", "Mercado (L)": "{:.0%}"}),
                width="stretch", hide_index=True, height=400,
            )
        st.caption("El mercado es muy eficiente: una discrepancia grande suele delatar un dato "
                   "que le falta al modelo, no una oportunidad real.")
