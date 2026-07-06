"""Corre el Monte Carlo del torneo y guarda las probabilidades por seleccion.

    uv run python -m scripts.build_simulation [--sims N]

Consume lo que dejaron las fases previas (modelo de goles, fixture, Elo, cuadro)
y produce data/processed/simulation_probabilities.parquet con, por equipo, la
probabilidad de ganar el grupo, clasificar, llegar a cada ronda y ser campeon.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model import markets as mk
from model.goals import FittedGoalModel
from tournament import format2026, montecarlo

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "reference"


def load_fitted_model() -> FittedGoalModel:
    """Reconstruye el modelo de goles desde los parametros guardados (Fase 2)."""
    strengths = pd.read_parquet(PROCESSED_DIR / "goal_model_strengths.parquet")
    params = json.loads((PROCESSED_DIR / "goal_model_params.json").read_text(encoding="utf-8"))
    return FittedGoalModel(
        intercept=params["intercept"],
        home_advantage=params["home_advantage"],
        rho=params["rho"],
        attack=dict(zip(strengths["team"], strengths["attack"])),
        defense=dict(zip(strengths["team"], strengths["defense"])),
        meta=params.get("meta", {}),
    )


def build_match_markets(model: FittedGoalModel, fixture: pd.DataFrame) -> pd.DataFrame:
    """Precomputa los mercados de la Capa 1 de cada partido de fase de grupos.

    La app solo lee este Parquet: la matematica del modelo no se ejecuta en la
    capa de presentacion. La localia se aplica al anfitrion (aunque sea el equipo
    visitante del calendario), reorientando la matriz cuando hace falta.
    """
    goals_axis = np.arange(11)
    rows = []
    for r in fixture.itertuples(index=False):
        adv = montecarlo._host_with_advantage(r.home_team, r.away_team, r.country)
        if adv == r.away_team:
            # El anfitrion es el visitante: matriz en su marco y transpuesta.
            matrix = model.score_matrix(r.away_team, r.home_team, home_advantage=True).T
        else:
            matrix = model.score_matrix(r.home_team, r.away_team, home_advantage=(adv == r.home_team))
        bundle = mk.all_markets(matrix)
        rows.append({
            "date": r.date, "group": r.group,
            "home_team": r.home_team, "away_team": r.away_team,
            "played": bool(r.played), "home_score": r.home_score, "away_score": r.away_score,
            "lambda_home": float((matrix.sum(axis=1) * goals_axis).sum()),
            "lambda_away": float((matrix.sum(axis=0) * goals_axis).sum()),
            "p_home": bundle["result"]["home"], "p_draw": bundle["result"]["draw"], "p_away": bundle["result"]["away"],
            "over_1_5": bundle["over_under"]["1.5"]["over"],
            "over_2_5": bundle["over_under"]["2.5"]["over"],
            "over_3_5": bundle["over_under"]["3.5"]["over"],
            "btts_yes": bundle["btts"]["yes"],
            "cs_home": bundle["clean_sheet"]["home"], "cs_away": bundle["clean_sheet"]["away"],
            "exact_scores": json.dumps(bundle["exact_score"]),
        })
    return pd.DataFrame(rows)


def main(n_sims: int) -> None:
    model = load_fitted_model()
    fixture = pd.read_parquet(PROCESSED_DIR / "fixture_group_stage.parquet")
    bracket = format2026.load_bracket(REFERENCE_DIR)
    # Elo como proxy del ranking FIFA para el desempate final de grupo.
    elo = pd.read_parquet(PROCESSED_DIR / "elo_current.parquet")
    final_key = dict(zip(elo["team"], elo["rating"]))

    # Cruces de eliminatoria ya jugados: se fijan (capa viva del cuadro).
    played_knockout = {}
    ko_path = PROCESSED_DIR / "knockout_results.parquet"
    if ko_path.exists():
        ko = pd.read_parquet(ko_path)
        ko = ko[ko["winner"].notna()]
        played_knockout = {frozenset((r.home_team, r.away_team)): r.winner
                           for r in ko.itertuples(index=False)}

    played = int(fixture["played"].sum())
    print(f"Corriendo {n_sims:,} simulaciones del torneo "
          f"({played} de grupos + {len(played_knockout)} de eliminatoria fijados)...")
    probs = montecarlo.run_monte_carlo(
        fixture, model, bracket, n_sims=n_sims, final_key=final_key, seed=0,
        played_knockout=played_knockout,
    )

    probs.to_parquet(PROCESSED_DIR / "simulation_probabilities.parquet", index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sims": n_sims,
        "fixture_played": played,
        "knockout_played": len(played_knockout),
    }
    (PROCESSED_DIR / "simulation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Mercados de la Capa 1 por partido, para el dashboard.
    match_markets = build_match_markets(model, fixture)
    match_markets.to_parquet(PROCESSED_DIR / "match_markets.parquet", index=False)
    print(f"Mercados por partido guardados: {len(match_markets)} partidos")

    pct = lambda x: f"{x:6.1%}"
    print("\nProbabilidad de ser campeon (top 15):")
    for i, row in probs.head(15).iterrows():
        print(f"  {i + 1:>2}. {row['team']:<20} {pct(row['champion'])}  "
              f"(final {pct(row['final'])} | clasifica {pct(row['qualify'])})")

    print("\nClasificacion esperada de los anfitriones:")
    for host in ["Mexico", "United States", "Canada"]:
        r = probs[probs["team"] == host]
        if not r.empty:
            r = r.iloc[0]
            print(f"  {host:<16} gana grupo {pct(r['win_group'])} | "
                  f"clasifica {pct(r['qualify'])} | campeon {pct(r['champion'])}")
    print("\nListo. Probabilidades en data/processed/simulation_probabilities.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo del torneo.")
    parser.add_argument("--sims", type=int, default=10_000, help="Numero de simulaciones.")
    args = parser.parse_args()
    main(args.sims)
