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

import pandas as pd

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


def main(n_sims: int) -> None:
    model = load_fitted_model()
    fixture = pd.read_parquet(PROCESSED_DIR / "fixture_group_stage.parquet")
    bracket = format2026.load_bracket(REFERENCE_DIR)
    # Elo como proxy del ranking FIFA para el desempate final de grupo.
    elo = pd.read_parquet(PROCESSED_DIR / "elo_current.parquet")
    final_key = dict(zip(elo["team"], elo["rating"]))

    played = int(fixture["played"].sum())
    print(f"Corriendo {n_sims:,} simulaciones del torneo "
          f"({played} partidos ya jugados se fijan, {len(fixture) - played} se simulan)...")
    probs = montecarlo.run_monte_carlo(
        fixture, model, bracket, n_sims=n_sims, final_key=final_key, seed=0
    )

    probs.to_parquet(PROCESSED_DIR / "simulation_probabilities.parquet", index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sims": n_sims,
        "fixture_played": played,
    }
    (PROCESSED_DIR / "simulation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
