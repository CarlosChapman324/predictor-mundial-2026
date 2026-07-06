"""Recalculo de la capa viva: actualiza todo y guarda una foto con timestamp.

El Mundial ya esta en marcha. Cada vez que se juegan partidos, este script:
  1. re-descarga el historico (que trae los resultados nuevos),
  2. reconstruye datos y Elo, reajusta el modelo de goles,
  3. re-simula el torneo condicionado a lo ya jugado,
  4. guarda una FOTO de las probabilidades con su timestamp en el historico,
     y ancla (una sola vez) el pronostico SIN condicionar como referencia.

Asi se puede mostrar como se mueven las probabilidades a lo largo del torneo.

    uv run python -m scripts.refresh                 # descarga datos frescos
    uv run python -m scripts.refresh --no-download    # reusa el CSV cacheado

Pensado para correrse a diario (cron). El eje del historico es "partidos
jugados": 0 = pronostico sin condicionar, y va creciendo con el torneo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import build_data, build_model, build_simulation
from tournament import format2026, montecarlo

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "reference"
HISTORY_PATH = PROCESSED_DIR / "predictions_history.parquet"


def append_snapshot(probs: pd.DataFrame, matches_played: int, generated_at: str, label: str) -> pd.DataFrame:
    """Anade (o reemplaza) la foto de un nivel de 'partidos jugados' al historico."""
    snapshot = probs[["team", "champion", "qualify"]].copy()
    snapshot["matches_played"] = matches_played
    snapshot["generated_at"] = generated_at
    snapshot["label"] = label
    if HISTORY_PATH.exists():
        history = pd.read_parquet(HISTORY_PATH)
        history = history[history["matches_played"] != matches_played]  # reemplaza misma foto
        snapshot = pd.concat([history, snapshot], ignore_index=True)
    snapshot = snapshot.sort_values(["matches_played", "champion"], ascending=[True, False]).reset_index(drop=True)
    snapshot.to_parquet(HISTORY_PATH, index=False)
    return snapshot


def main(download: bool = True, n_sims: int = 10_000) -> None:
    # 1-3. Recalculo completo del pipeline.
    build_data.main(download=download)
    build_model.main()
    build_simulation.main(n_sims)

    # 4. Fotos del historico.
    model = build_simulation.load_fitted_model()
    fixture = pd.read_parquet(PROCESSED_DIR / "fixture_group_stage.parquet")
    bracket = format2026.load_bracket(REFERENCE_DIR)
    elo = pd.read_parquet(PROCESSED_DIR / "elo_current.parquet")
    final_key = dict(zip(elo["team"], elo["rating"]))
    current = pd.read_parquet(PROCESSED_DIR / "simulation_probabilities.parquet")
    # El eje del historico cuenta TODO lo jugado: fase de grupos + eliminatoria.
    ko_path = PROCESSED_DIR / "knockout_results.parquet"
    ko_played = int(len(pd.read_parquet(ko_path))) if ko_path.exists() else 0
    played = int(fixture["played"].sum()) + ko_played
    now = datetime.now(timezone.utc).isoformat()

    # Ancla "sin condicionar" (partidos jugados = 0), congelada tras la primera vez.
    has_baseline = HISTORY_PATH.exists() and 0 in set(pd.read_parquet(HISTORY_PATH)["matches_played"])
    if not has_baseline:
        baseline = montecarlo.run_monte_carlo(
            fixture, model, bracket, n_sims=n_sims, final_key=final_key, seed=0, ignore_played=True
        )
        append_snapshot(baseline, 0, now, "Sin condicionar")
        print("Ancla 'sin condicionar' guardada en el historico.")

    append_snapshot(current, played, now, f"{played} jugados")
    print(f"\nFoto guardada: {played} partidos jugados @ {now[:19]}")

    _print_movement(played)
    print("\nListo. Historico en data/processed/predictions_history.parquet")


def _print_movement(played: int) -> None:
    """Muestra que selecciones subieron o bajaron frente al pronostico sin condicionar."""
    history = pd.read_parquet(HISTORY_PATH)
    if played == 0 or 0 not in set(history["matches_played"]):
        return
    base = history[history["matches_played"] == 0].set_index("team")["champion"]
    cur = history[history["matches_played"] == played].set_index("team")["champion"]
    delta = (cur - base).dropna().sort_values()
    if delta.empty:
        return
    print("\nMovimiento de la probabilidad de campeon (vs sin condicionar):")
    movers = pd.concat([delta.tail(4).iloc[::-1], delta.head(4)])
    for team, d in movers.items():
        arrow = "sube" if d > 0 else "baja"
        print(f"  {team:<18} {cur[team]:6.1%}  ({arrow} {abs(d) * 100:.1f} pp)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculo de la capa viva.")
    parser.add_argument("--no-download", action="store_true", help="Reusa el CSV cacheado.")
    parser.add_argument("--sims", type=int, default=10_000, help="Numero de simulaciones.")
    args = parser.parse_args()
    main(download=not args.no_download, n_sims=args.sims)
