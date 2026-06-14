"""Compara las probabilidades del modelo con las del mercado de apuestas.

    THE_ODDS_API_KEY=tu_clave uv run python -m scripts.build_market
    # o, sin clave, deja un CSV en data/raw/odds_2026.csv y corre:
    uv run python -m scripts.build_market

Fuente de cuotas (en este orden):
  1. The Odds API, si la variable de entorno THE_ODDS_API_KEY esta definida.
  2. data/raw/odds_2026.csv (columnas: home_team, away_team, home_odds, draw_odds, away_odds).
Si no hay ninguna, explica como conseguirlas y termina sin inventar nada.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market import ingest, odds
from market.odds import OUTCOMES
from model.markets import one_x_two
from scripts.build_simulation import load_fitted_model

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

HOSTS = {"United States", "Canada", "Mexico"}
# Nombres de The Odds API que difieren de los del dataset historico.
TEAM_ALIASES = {"USA": "United States", "IR Iran": "Iran", "Korea Republic": "South Korea",
                "Czechia": "Czech Republic", "Turkiye": "Turkey", "Cote d'Ivoire": "Ivory Coast",
                "Bosnia & Herzegovina": "Bosnia and Herzegovina"}


def _canonical(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def model_1x2(model, home: str, away: str) -> list[float]:
    """1X2 del modelo en la orientacion (home, away), con localia si juega un anfitrion."""
    if home in HOSTS and away not in HOSTS:
        r = one_x_two(model.score_matrix(home, away, home_advantage=True))
        return [r["home"], r["draw"], r["away"]]
    if away in HOSTS and home not in HOSTS:
        r = one_x_two(model.score_matrix(away, home, home_advantage=True))
        return [r["away"], r["draw"], r["home"]]  # reorientado a (home, away)
    r = one_x_two(model.score_matrix(home, away, home_advantage=False))
    return [r["home"], r["draw"], r["away"]]


def get_odds():
    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return ingest.fetch_the_odds_api(key), "The Odds API"
    csv_path = RAW_DIR / "odds_2026.csv"
    if csv_path.exists():
        return ingest.load_odds_csv(csv_path), str(csv_path.relative_to(ROOT))
    return None, None


def _print_instructions():
    print("No hay cuotas disponibles, asi que no se genera la comparacion (no se inventan datos).")
    print("\nPara obtener una comparacion real, elige UNA opcion:")
    print("  1) API: consigue una clave gratis en https://the-odds-api.com y corre:")
    print("       THE_ODDS_API_KEY=tu_clave uv run python -m scripts.build_market")
    print("  2) CSV: crea data/raw/odds_2026.csv con columnas:")
    print("       home_team, away_team, home_odds, draw_odds, away_odds  (cuotas decimales)")


def main() -> None:
    odds_df, source = get_odds()
    if odds_df is None or odds_df.empty:
        _print_instructions()
        return

    print(f"Cuotas obtenidas de: {source} ({len(odds_df)} filas de casas)")
    consensus = ingest.consensus_market(odds_df)
    consensus["home_team"] = consensus["home_team"].map(_canonical)
    consensus["away_team"] = consensus["away_team"].map(_canonical)

    model = load_fitted_model()
    known = set(model.attack)
    rows = []
    for r in consensus.itertuples(index=False):
        if r.home_team not in known or r.away_team not in known:
            continue  # equipo no modelado: lo dejamos fuera y lo reportamos
        mh, md, ma = model_1x2(model, r.home_team, r.away_team)
        rows.append({
            "home_team": r.home_team, "away_team": r.away_team,
            "model_home": mh, "model_draw": md, "model_away": ma,
            "market_home": r.market_home, "market_draw": r.market_draw, "market_away": r.market_away,
            "overround": r.overround,
        })
    comparison = odds.add_edges(pd.DataFrame(rows))

    # Resultados reales (capa viva) para comparar quien predijo mejor, si los hay.
    outcome_col = None
    fixture_path = PROCESSED_DIR / "fixture_group_stage.parquet"
    if fixture_path.exists():
        comparison = _attach_outcomes(comparison, pd.read_parquet(fixture_path))
        outcome_col = "outcome"

    comparison.to_parquet(PROCESSED_DIR / "market_comparison.parquet", index=False)
    summary = odds.efficiency_summary(comparison, outcome_col=outcome_col)
    (PROCESSED_DIR / "market_meta.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "source": source, **summary}, indent=2),
        encoding="utf-8",
    )

    unmatched = len(consensus) - len(comparison)
    print(f"Partidos comparados: {len(comparison)}" + (f" ({unmatched} sin emparejar por nombre)" if unmatched else ""))
    print(f"Margen medio de las casas (overround): {summary['avg_overround']:.1%}")
    print(f"Coinciden en el favorito: {summary['agreement_favorite']:.0%} de los partidos")
    print(f"Correlacion modelo vs mercado: {summary['prob_correlation']:.3f}")
    if "rps_model" in summary:
        print(f"En {summary['n_settled']} partidos ya jugados: RPS modelo {summary['rps_model']:.4f} | "
              f"mercado {summary['rps_market']:.4f}")

    print("\nMayores discrepancias (donde el modelo mas se aparta del mercado):")
    top = comparison.reindex(comparison["max_abs_edge"].sort_values(ascending=False).index).head(8)
    for r in top.itertuples(index=False):
        print(f"  {r.home_team} vs {r.away_team}: modelo "
              f"{r.model_home:.0%}/{r.model_draw:.0%}/{r.model_away:.0%} | mercado "
              f"{r.market_home:.0%}/{r.market_draw:.0%}/{r.market_away:.0%}")
    print("\nNota: el mercado es muy eficiente; las discrepancias grandes suelen senalar "
          "un dato que le falta al modelo, no una oportunidad real.")
    print("\nListo. Comparacion en data/processed/market_comparison.parquet")


def _attach_outcomes(comparison: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    """Une el resultado real (0/1/2) de los partidos ya jugados, por par de equipos."""
    played = fixture[fixture["played"]].copy()
    result = {}
    for r in played.itertuples(index=False):
        out = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        result[frozenset((r.home_team, r.away_team))] = (r.home_team, out)
    outcomes = []
    for r in comparison.itertuples(index=False):
        key = frozenset((r.home_team, r.away_team))
        if key in result:
            fixture_home, out = result[key]
            # Reorienta el resultado a la orientacion (home, away) de la comparacion.
            if fixture_home != r.home_team and out in (0, 2):
                out = 2 - out
            outcomes.append(out)
        else:
            outcomes.append(None)
    comparison = comparison.copy()
    comparison["outcome"] = outcomes
    return comparison


if __name__ == "__main__":
    main()
