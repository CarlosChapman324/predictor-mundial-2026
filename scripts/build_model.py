"""Ajusta el modelo de goles sobre el historico y guarda los parametros.

    uv run python -m scripts.build_model

Guarda en data/processed/:
  - goal_model_strengths.parquet : ataque y defensa por seleccion
  - goal_model_params.json       : intercepto, localia, rho y metadatos

Tambien imprime un par de partidos de ejemplo con sus mercados, como sanity check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model import goals, markets

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Anfitriones: solo ellos tienen localia en el Mundial (juegan en su pais).
HOST_TEAMS = {"United States", "Canada", "Mexico"}


def _print_match(model: goals.FittedGoalModel, home: str, away: str, *, home_advantage: bool):
    lh, la = model.expected_goals(home, away, home_advantage=home_advantage)
    matrix = model.score_matrix(home, away, home_advantage=home_advantage)
    r = markets.one_x_two(matrix)
    ou = markets.over_under(matrix, 2.5)
    btts = markets.both_teams_to_score(matrix)
    top = markets.exact_score(matrix, 1)[0]
    sede = " (con localia)" if home_advantage else " (cancha neutral)"
    print(f"\n  {home} vs {away}{sede}")
    print(f"    Goles esperados: {lh:.2f} - {la:.2f}")
    print(f"    1X2: {home[:12]} {r['home']:.0%} | Empate {r['draw']:.0%} | {away[:12]} {r['away']:.0%}")
    print(f"    Over 2.5: {ou['over']:.0%} | BTTS si: {btts['yes']:.0%} | "
          f"Marcador top: {top['home_goals']}-{top['away_goals']} ({top['prob']:.0%})")


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    print(f"Ajustando el modelo de goles sobre {len(matches):,} partidos...")

    model = goals.fit_goal_model(matches)
    print(f"  Convergio: {model.meta['converged']} | "
          f"partidos en la ventana: {model.meta['n_matches']:,} | "
          f"selecciones: {model.meta['n_teams']}")
    print(f"  Intercepto: {model.intercept:.3f} | "
          f"Localia: {model.home_advantage:.3f} | rho: {model.rho:.3f}")

    # Guardar fuerzas y parametros.
    strengths = (
        pd.DataFrame({
            "team": model.teams,
            "attack": [model.attack[t] for t in model.teams],
            "defense": [model.defense[t] for t in model.teams],
        })
        .sort_values("attack", ascending=False)
        .reset_index(drop=True)
    )
    strengths.to_parquet(PROCESSED_DIR / "goal_model_strengths.parquet", index=False)
    params = {
        "intercept": model.intercept,
        "home_advantage": model.home_advantage,
        "rho": model.rho,
        "meta": model.meta,
    }
    (PROCESSED_DIR / "goal_model_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Sanity check 1: mejores ataques y mejores defensas.
    print("\nTop 10 ataque (marca mucho):")
    for _, row in strengths.head(10).iterrows():
        print(f"  {row['team']:<22} {row['attack']:+.3f}")
    print("\nTop 10 defensa (recibe poco, defensa mas negativa):")
    best_def = strengths.sort_values("defense").head(10)
    for _, row in best_def.iterrows():
        print(f"  {row['team']:<22} {row['defense']:+.3f}")

    # Sanity check 2: un par de partidos concretos.
    print("\nPartidos de ejemplo:")
    _print_match(model, "Spain", "Cape Verde", home_advantage=False)
    _print_match(model, "Mexico", "South Africa", home_advantage=True)
    _print_match(model, "Brazil", "Scotland", home_advantage=False)
    print("\nListo. Parametros del modelo en data/processed/.")


if __name__ == "__main__":
    main()
