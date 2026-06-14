"""Backtesting del modelo contra torneos pasados.

Para cada torneo de prueba se entrena el modelo SOLO con partidos anteriores a su
fecha de inicio (corte temporal estricto: nunca se usan datos del futuro) y se
predicen sus partidos. Se compara contra dos baselines:
  - uniforme: 1/3, 1/3, 1/3 (el azar; el piso que hay que superar);
  - Elo: prediccion por fuerza (ranking Elo a la fecha del corte), un baseline
    simple pero serio (el equivalente honesto a 'predecir por ranking FIFA').

El objetivo del portafolio: mostrar con metricas formales que el modelo le gana
al azar y se acerca o supera al baseline de fuerza, y que esta bien calibrado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import elo, goals, markets

# Torneos de prueba: Mundiales y Eurocopas recientes. (Euro 2020 se jugo en 2021).
TEST_TOURNAMENTS = [
    {"name": "FIFA World Cup", "year": 2014, "label": "Mundial 2014"},
    {"name": "FIFA World Cup", "year": 2018, "label": "Mundial 2018"},
    {"name": "FIFA World Cup", "year": 2022, "label": "Mundial 2022"},
    {"name": "UEFA Euro", "year": 2016, "label": "Euro 2016"},
    {"name": "UEFA Euro", "year": 2021, "label": "Euro 2020"},
    {"name": "UEFA Euro", "year": 2024, "label": "Euro 2024"},
]

# Constante del baseline Elo: probabilidad de empate, maxima en partidos parejos
# y decreciente con la diferencia de fuerza. Es una heuristica transparente.
ELO_DRAW_C = 0.32


def outcome_index(home_score: int, away_score: int) -> int:
    """0 gana local, 1 empate, 2 gana visitante."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def model_probs(model: goals.FittedGoalModel, home, away, neutral: bool) -> list[float]:
    """1X2 del modelo de goles (con localia salvo cancha neutral)."""
    matrix = model.score_matrix(home, away, home_advantage=not neutral)
    r = markets.one_x_two(matrix)
    return [r["home"], r["draw"], r["away"]]


def elo_probs(ratings: dict, home, away, neutral: bool, base: float = elo.BASE_RATING) -> list[float]:
    """Baseline 1X2 derivado solo del Elo a la fecha del corte."""
    r_home = ratings.get(home, base)
    r_away = ratings.get(away, base)
    advantage = 0.0 if neutral else elo.HOME_ADVANTAGE
    we = elo.expected_score(r_home + advantage - r_away)
    p_draw = ELO_DRAW_C * (1.0 - abs(2.0 * we - 1.0))
    return [(1.0 - p_draw) * we, p_draw, (1.0 - p_draw) * (1.0 - we)]


def tournament_matches(matches: pd.DataFrame, name: str, year: int) -> pd.DataFrame:
    """Partidos de un torneo concreto (por nombre y ano de disputa)."""
    dates = pd.to_datetime(matches["date"])
    mask = (matches["tournament"] == name) & (dates.dt.year == year)
    return matches.loc[mask].sort_values("date")


def backtest_tournament(matches: pd.DataFrame, name: str, year: int, *, fit_kwargs=None) -> dict:
    """Entrena con datos anteriores al torneo y predice sus partidos.

    Devuelve un dict con las predicciones (modelo y baselines) y los resultados.
    """
    fit_kwargs = fit_kwargs or {}
    test = tournament_matches(matches, name, year)
    if test.empty:
        raise ValueError(f"No hay partidos para {name} {year} en el historico")
    cutoff = pd.to_datetime(test["date"]).min()

    train = matches.loc[pd.to_datetime(matches["date"]) < cutoff]
    model = goals.fit_goal_model(train, reference_date=cutoff, **fit_kwargs)
    _, elo_current = elo.compute_elo(train)
    ratings = dict(zip(elo_current["team"], elo_current["rating"]))

    preds = {"model": [], "uniform": [], "elo": []}
    outcomes = []
    for row in test.itertuples(index=False):
        neutral = bool(row.neutral)
        outcomes.append(outcome_index(row.home_score, row.away_score))
        preds["model"].append(model_probs(model, row.home_team, row.away_team, neutral))
        preds["uniform"].append([1 / 3, 1 / 3, 1 / 3])
        preds["elo"].append(elo_probs(ratings, row.home_team, row.away_team, neutral))

    return {
        "label": next((t["label"] for t in TEST_TOURNAMENTS
                       if t["name"] == name and t["year"] == year), f"{name} {year}"),
        "outcomes": np.asarray(outcomes, dtype=int),
        "predictions": {k: np.asarray(v, dtype=float) for k, v in preds.items()},
        "cutoff": str(cutoff.date()),
        "n_train": int(len(train)),
    }


def run_backtest(matches: pd.DataFrame, tournaments=None, *, fit_kwargs=None):
    """Corre el backtesting sobre todos los torneos de prueba.

    Devuelve (summary, calibration, pooled):
      - summary: una fila por (torneo, predictor) con n, rps, log_loss, brier.
      - calibration: tabla de calibracion del modelo, agregada sobre todo.
      - pooled: predicciones y resultados de todos los torneos juntos.
    """
    from validation import metrics

    tournaments = tournaments or TEST_TOURNAMENTS
    rows = []
    pooled_outcomes = []
    pooled_preds = {"model": [], "uniform": [], "elo": []}

    for t in tournaments:
        result = backtest_tournament(matches, t["name"], t["year"], fit_kwargs=fit_kwargs)
        outcomes = result["outcomes"]
        for predictor, preds in result["predictions"].items():
            stats = metrics.summarize(preds, outcomes)
            rows.append({"tournament": result["label"], "predictor": predictor, **stats})
            pooled_preds[predictor].append(preds)
        pooled_outcomes.append(outcomes)

    pooled_outcomes = np.concatenate(pooled_outcomes)
    pooled_preds = {k: np.concatenate(v) for k, v in pooled_preds.items()}
    for predictor, preds in pooled_preds.items():
        stats = metrics.summarize(preds, pooled_outcomes)
        rows.append({"tournament": "Todos", "predictor": predictor, **stats})

    summary = pd.DataFrame(rows)
    calibration = metrics.calibration_table(pooled_preds["model"], pooled_outcomes)
    pooled = {"outcomes": pooled_outcomes, "predictions": pooled_preds}
    return summary, calibration, pooled
