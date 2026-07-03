"""Predicho (desde antes del torneo) vs lo que realmente paso, fase de grupos.

Para responder honestamente "que dijo el modelo desde un principio y que ocurrio":
se reentrena el modelo de goles usando SOLO datos anteriores al inicio del Mundial
(11 de junio de 2026), sin fuga de informacion, y se predice el 1X2 de cada partido
de fase de grupos. Luego se compara contra el resultado real.

    uv run python -m scripts.build_predicted_vs_actual

Guarda data/processed/predicted_vs_actual.parquet (una fila por partido jugado).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import goals, markets
from tournament.montecarlo import _host_with_advantage
from validation.metrics import ranked_probability_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TOURNAMENT_START = pd.Timestamp("2026-06-11")


def _predict_1x2(model, home, away, country):
    """1X2 del modelo en la orientacion (home, away), con localia si juega un anfitrion."""
    adv = _host_with_advantage(home, away, country)
    if adv == away:
        matrix = model.score_matrix(away, home, home_advantage=True).T  # reorienta a (home, away)
    else:
        matrix = model.score_matrix(home, away, home_advantage=(adv == home))
    r = markets.one_x_two(matrix)
    return r["home"], r["draw"], r["away"]


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    # Modelo entrenado SOLO con el pasado (corte estricto antes del torneo).
    train = matches[matches["date"] < TOURNAMENT_START]
    model = goals.fit_goal_model(train, reference_date=TOURNAMENT_START)
    print(f"Modelo pre-torneo entrenado con {len(train):,} partidos anteriores al 2026-06-11.")

    fixture = pd.read_parquet(PROCESSED_DIR / "fixture_group_stage.parquet")
    played = fixture[fixture["played"]].copy()

    rows = []
    for r in played.itertuples(index=False):
        p_home, p_draw, p_away = _predict_1x2(model, r.home_team, r.away_team, r.country)
        probs = [p_home, p_draw, p_away]
        outcome = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        pick = int(np.argmax(probs))
        rows.append({
            "group": r.group, "date": r.date,
            "home_team": r.home_team, "away_team": r.away_team,
            "home_score": int(r.home_score), "away_score": int(r.away_score),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "actual_outcome": outcome, "model_pick": pick,
            "hit": pick == outcome, "prob_actual": probs[outcome],
        })
    out = pd.DataFrame(rows)
    out.to_parquet(PROCESSED_DIR / "predicted_vs_actual.parquet", index=False)

    accuracy = out["hit"].mean()
    avg_prob = out["prob_actual"].mean()
    rps = np.mean([ranked_probability_score([r.p_home, r.p_draw, r.p_away], r.actual_outcome)
                   for r in out.itertuples(index=False)])
    (PROCESSED_DIR / "predicted_vs_actual_meta.json").write_text(
        json.dumps({"accuracy": float(accuracy), "avg_prob_actual": float(avg_prob),
                    "rps": float(rps), "n": int(len(out))}, indent=2), encoding="utf-8")
    print(f"\nPredicho (pre-torneo) vs real, {len(out)} partidos de grupos:")
    print(f"  Aciertos del favorito 1X2: {accuracy:.0%} ({int(out['hit'].sum())}/{len(out)})")
    print(f"  Probabilidad media que el modelo dio a lo que paso: {avg_prob:.1%}")
    print(f"  RPS en el torneo real: {rps:.4f} (menor es mejor; ~0.20 es buen nivel)")
    print("\nMayores sorpresas (lo que paso, que el modelo veia poco probable):")
    for r in out.sort_values("prob_actual").head(5).itertuples(index=False):
        res = f"{r.home_score}-{r.away_score}"
        print(f"  {r.home_team} vs {r.away_team}: {res} (el modelo le daba {r.prob_actual:.0%})")
    print("\nListo. Comparacion en data/processed/predicted_vs_actual.parquet")


if __name__ == "__main__":
    main()
