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
_ko_fixed = meta.get("knockout_played", 0)
st.caption(
    f"{meta.get('n_sims', 0):,} simulaciones Monte Carlo &middot; "
    f"{meta.get('fixture_played', 0)} partidos de grupos"
    + (f" + {_ko_fixed} cruces de eliminatoria fijados" if _ko_fixed else " fijados")
)


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


(tab_champ, tab_evo, tab_groups, tab_ko, tab_path, tab_match, tab_val, tab_real,
 tab_market, tab_scorers, tab_value) = st.tabs(
    ["Campeon", "Evolucion", "Grupos", "Eliminatorias", "Camino al titulo", "Por partido",
     "Validacion", "Predicho vs Real", "Mercado", "Goleadores", "Valor"]
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


# --- Evolucion -------------------------------------------------------------
with tab_evo:
    history = data.predictions_history()
    if history is None or history.empty:
        st.info("Aun no hay historico. Corre `uv run python -m scripts.refresh` tras cada "
                "jornada para registrar como se mueven las probabilidades.")
    elif history["matches_played"].nunique() < 2:
        latest = int(history["matches_played"].max())
        st.info(f"Solo hay una foto ({latest} partidos jugados). El grafico se llena al "
                "recalcular tras jugarse mas partidos: `uv run python -m scripts.refresh`.")
    else:
        st.plotly_chart(charts.probability_evolution(history, n=8), width="stretch")
        st.caption("Cada foto se guarda con timestamp al recalcular. El punto en 0 es el "
                   "pronostico sin condicionar; las lineas se mueven al fijar resultados reales.")


# --- Grupos ----------------------------------------------------------------
with tab_groups:
    groups = data.groups()
    standings = data.group_standings()
    if not groups:
        st.info("Falta groups.json.")
    else:
        choice = st.selectbox("Grupo", list(groups.keys()), format_func=lambda g: f"Grupo {g}")
        sg = standings[standings["group"] == choice] if standings is not None else None

        if sg is not None and not sg.empty and sg["played"].sum() > 0:
            # Fase de grupos con resultados reales: tabla de posiciones.
            complete = bool((standings["played"] == 3).all())
            t = sg.sort_values("position").copy()
            if complete:
                t["Estado"] = np.where(
                    t["qualified"] & (t["position"] <= 2), "Clasificado",
                    np.where(t["qualified"], "Clasificado (mejor 3o)", "Eliminado"),
                )
            show = t[["position", "team", "played", "points", "gf", "ga", "gd"]
                     + (["Estado"] if complete else [])]
            show.columns = ["Pos", "Seleccion", "PJ", "Pts", "GF", "GC", "DG"] + (
                ["Estado"] if complete else [])
            st.dataframe(show, width="stretch", hide_index=True)
            mm = data.match_markets()
            if mm is not None:
                res = mm[(mm["group"] == choice) & mm["played"]].sort_values("date")
                if not res.empty:
                    st.markdown("**Resultados del grupo**")
                    lines = "  \n".join(
                        f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}"
                        for r in res.itertuples(index=False)
                    )
                    st.markdown(lines)
            st.caption("Tabla real con los desempates 2026 (head-to-head antes que la diferencia "
                       "global). Clasifican el 1o y el 2o, mas los 8 mejores terceros de los 12.")
        else:
            # Pre-torneo: proyeccion probabilistica del grupo.
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


# --- Eliminatorias ----------------------------------------------------------
def _knockout_round(d) -> str:
    """Ronda de un cruce del 2026 segun su fecha (calendario oficial)."""
    if d.month == 6 or d.day <= 3:
        return "Dieciseisavos"
    if d.day <= 8:
        return "Octavos"
    if d.day <= 12:
        return "Cuartos"
    if d.day <= 16:
        return "Semifinales"
    return "Tercer puesto" if d.day == 18 else "Final"


with tab_ko:
    ko = data.knockout_results()
    if ko is None or ko.empty:
        st.info("La eliminatoria aun no arranca: los cruces apareceran aqui al jugarse.")
    else:
        standings = data.group_standings()
        # En eliminacion directa, perder un cruce te saca (el perdedor de cada partido).
        losers = {
            r.away_team if r.winner == r.home_team else r.home_team
            for r in ko.itertuples(index=False) if r.winner is not None
        }
        if standings is not None:
            qualified = set(standings[standings["qualified"]]["team"])
        else:
            qualified = set(sim["team"])
        alive = qualified - losers

        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi("Cruces jugados", f"{len(ko)}", "de 31 del cuadro"), unsafe_allow_html=True)
        c2.markdown(theme.kpi("Siguen vivos", f"{len(alive)}", "de 32 clasificados", accent=True), unsafe_allow_html=True)
        favorito = sim[sim["team"].isin(alive)].nlargest(1, "champion")
        if not favorito.empty:
            f = favorito.iloc[0]
            c3.markdown(theme.kpi(f["team"], pct(f["champion"]), "favorito al titulo"), unsafe_allow_html=True)

        st.write("")
        left, right = st.columns([3, 2])
        with left:
            ko = ko.sort_values("date")
            ko["round"] = ko["date"].map(_knockout_round)
            order = ["Dieciseisavos", "Octavos", "Cuartos", "Semifinales", "Tercer puesto", "Final"]
            for round_name in [r for r in order if r in set(ko["round"])]:
                st.markdown(f"**{round_name}**")
                lines = []
                for r in ko[ko["round"] == round_name].itertuples(index=False):
                    home = f"**{r.home_team}**" if r.winner == r.home_team else r.home_team
                    away = f"**{r.away_team}**" if r.winner == r.away_team else r.away_team
                    marcador = f"{int(r.home_score)}-{int(r.away_score)}"
                    linea = f"{home} {marcador} {away}"
                    if r.home_score == r.away_score:
                        linea += f" &middot; {r.winner} por penales"
                    lines.append(linea)
                st.markdown("  \n".join(lines), unsafe_allow_html=True)
        with right:
            vivos = sim[sim["team"].isin(alive)]
            if not vivos.empty:
                st.plotly_chart(charts.champion_bar(vivos, n=min(12, len(vivos))), width="stretch")
        st.caption("Cruces reales fijados en la simulacion (el ganador en negrita; los empates "
                   "los decidio la tanda de penales). Los eliminados quedan en 0% de campeon.")


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

        # Capa 2 (experimental): props del partido.
        props = data.props_predictions()
        if props is not None:
            pm = props[(props["home_team"] == row["home_team"]) & (props["away_team"] == row["away_team"])]
            if not pm.empty:
                pr = pm.iloc[0]
                st.divider()
                st.markdown("**Mercados de Capa 2** (experimental, confianza baja)")
                p1, p2, p3, p4 = st.columns(4)
                p1.markdown(theme.kpi("Corners", f"{pr['corners_total']:.1f}", f"Over 9.5: {pct(pr['corners_over_9_5'], 0)}"), unsafe_allow_html=True)
                p2.markdown(theme.kpi("Remates", f"{pr['shots_total']:.0f}", f"a puerta {pr['sot_total']:.0f}"), unsafe_allow_html=True)
                p3.markdown(theme.kpi("Tarjetas", f"{pr['cards_total']:.1f}", f"Over 4.5: {pct(pr['cards_over_4_5'], 0)}"), unsafe_allow_html=True)
                p4.markdown(theme.kpi("Roja", pct(pr["red_card_prob"], 0), "alguna en el partido"), unsafe_allow_html=True)
                st.caption("Arbitro del 2026 no disponible en el plan free: las tarjetas usan el promedio general.")

        # Indicador de valor (solo si el partido tiene cuotas).
        value = data.value_analysis()
        if value is not None:
            vm = value[(value["home_team"] == row["home_team"]) & (value["away_team"] == row["away_team"])]
            if not vm.empty:
                v = vm.iloc[0]
                pick = {"home": row["home_team"], "draw": "Empate", "away": row["away_team"]}[v["best_bet"]]
                st.markdown(f"**Valor vs mercado:** mayor EV en **{pick}** "
                            f"({v['best_ev']:+.0%}, confianza {v['confidence']}).")
                st.caption("El mercado es eficiente: un EV alto suele delatar un dato que le falta "
                           "al modelo, no una oportunidad real.")


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


# --- Goleadores (Capa 2, experimental) -------------------------------------
with tab_scorers:
    gb = data.golden_boot()
    if gb is None or gb.empty:
        st.info("Faltan los goleadores. Corre: uv run python -m scripts.build_scorers")
    else:
        st.warning("Capa 2 experimental (confianza baja): usa a los goleadores de los ultimos "
                   "4 anos como proxy del plantel; no incorpora convocatorias ni minutos jugados.")
        top, runner = gb.iloc[0], gb.iloc[1]
        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi(top["player"], pct(top["win_prob"]), f"Bota de Oro &middot; {top['team']}", accent=True), unsafe_allow_html=True)
        c2.markdown(theme.kpi("Goles esperados", f"{top['expected_goals']:.1f}", "del favorito en el torneo"), unsafe_allow_html=True)
        c3.markdown(theme.kpi(runner["player"], pct(runner["win_prob"]), f"2o &middot; {runner['team']}"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.golden_boot_bar(gb, n=12), width="stretch")
        with right:
            t = gb.head(15)[["player", "team", "expected_goals", "win_prob"]].copy()
            t.columns = ["Jugador", "Seleccion", "Goles esp.", "Bota de Oro"]
            st.dataframe(
                t.style.format({"Goles esp.": "{:.2f}", "Bota de Oro": "{:.1%}"}),
                width="stretch", hide_index=True, height=440,
            )
        ms = data.match_scorers()
        if ms is not None and not ms.empty:
            st.markdown("**Goleadores por partido**")
            ms = ms.copy()
            ms["match"] = ms["home_team"] + " vs " + ms["away_team"]
            pick = st.selectbox("Partido", sorted(ms["match"].unique()))
            sel = ms[ms["match"] == pick].sort_values("anytime", ascending=False).head(10)
            s = sel[["player", "player_team", "anytime"]].copy()
            s.columns = ["Jugador", "Seleccion", "Marca en algun momento"]
            st.dataframe(
                s.style.format({"Marca en algun momento": "{:.1%}"}),
                width="stretch", hide_index=True,
            )


# --- Valor -----------------------------------------------------------------
with tab_value:
    value = data.value_analysis()
    if value is None or value.empty:
        st.info("Falta el analisis de valor. Corre: uv run python -m scripts.build_value")
    else:
        n_value = int(value["has_value"].sum())
        media = value[value["has_value"] & (value["confidence"] == "media")]
        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi("Margen de las casas", pct(value["overround"].mean()), "overround medio"), unsafe_allow_html=True)
        c2.markdown(theme.kpi("EV positivo", f"{n_value}/{len(value)}", "casi todo: mercado eficiente"), unsafe_allow_html=True)
        c3.markdown(theme.kpi("Confianza media", f"{len(media)}", "EV moderado; el resto, longshots", accent=True), unsafe_allow_html=True)
        st.warning("El modelo ve 'valor' casi en todos lados porque es menos extremo que un mercado "
                   "eficiente. NO es un edge real: un EV alto suele delatar un dato que le falta al "
                   "modelo. Estudio de eficiencia de mercado, no un sistema de apuestas.")
        st.markdown("**Mayores discrepancias modelo vs mercado**")
        show = value.sort_values("best_ev", ascending=False).head(15)[
            ["home_team", "away_team", "best_bet", "best_ev", "confidence"]
        ].copy()
        show.columns = ["Local", "Visitante", "Mejor apuesta", "EV", "Confianza"]
        st.dataframe(
            show.style.format({"EV": "{:+.0%}"}),
            width="stretch", hide_index=True, height=420,
        )
        st.caption("Backtest de la estrategia: el motor esta listo y testeado; un backtest historico "
                   "necesitaria cuotas de seleccion de pago, asi que se alimenta de los partidos del "
                   "Mundial a medida que se juegan.")


# --- Predicho vs Real ------------------------------------------------------
with tab_real:
    pva = data.predicted_vs_actual()
    meta = data.predicted_vs_actual_meta() or {}
    if pva is None or pva.empty:
        st.info("Aun no hay comparacion. Corre: uv run python -m scripts.build_predicted_vs_actual")
    else:
        st.caption("Prediccion hecha con el modelo entrenado SOLO con datos anteriores al inicio del "
                   "torneo (11 jun 2026), sin fuga, y comparada con los resultados reales de la fase "
                   "de grupos. Es la prueba mas dura: el modelo contra el Mundial real, no un backtest.")
        acc = meta.get("accuracy", float(pva["hit"].mean()))
        avg = meta.get("avg_prob_actual", float(pva["prob_actual"].mean()))
        c1, c2, c3 = st.columns(3)
        c1.markdown(theme.kpi("Acierto 1X2", pct(acc, 0), f"{int(pva['hit'].sum())} de {len(pva)} partidos", accent=True), unsafe_allow_html=True)
        c2.markdown(theme.kpi("Prob. media en lo real", pct(avg), "que dio a lo que paso"), unsafe_allow_html=True)
        c3.markdown(theme.kpi("RPS en el torneo", f"{meta.get('rps', 0):.3f}", "menor es mejor (~0.20 buen nivel)"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([3, 2])
        with left:
            show = pva.copy()
            show["Partido"] = show["home_team"] + " vs " + show["away_team"]
            show["Modelo (L/E/V)"] = show.apply(
                lambda r: f"{r['p_home']:.0%} / {r['p_draw']:.0%} / {r['p_away']:.0%}", axis=1)
            show["Real"] = show["home_score"].astype(int).astype(str) + "-" + show["away_score"].astype(int).astype(str)
            show["Acierto"] = show["hit"].map({True: "Si", False: "No"})
            table = show[["Partido", "Modelo (L/E/V)", "Real", "prob_actual", "Acierto"]].rename(
                columns={"prob_actual": "Prob. a lo real"})
            st.dataframe(table.style.format({"Prob. a lo real": "{:.0%}"}),
                         width="stretch", hide_index=True, height=460)
        with right:
            st.markdown("**Mayores sorpresas** (lo que paso, que el modelo veia poco probable)")
            for r in pva.sort_values("prob_actual").head(8).itertuples(index=False):
                st.markdown(f"- **{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}** "
                            f"({r.prob_actual:.0%})")
