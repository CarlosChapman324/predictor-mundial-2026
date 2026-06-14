"""Submodelo de goleadores y Bota de Oro (Capa 2, experimental).

Idea (adelgazamiento de Poisson): si una seleccion mete en promedio lambda goles
y un jugador aporta una fraccion 'share' de los goles del equipo, sus goles
esperados en el partido son lambda * share, que se modelan como un Poisson. De
ahi:
  - Goleador en cualquier momento: P(marca >= 1) = 1 - exp(-lambda * share).
  - Bota de Oro: goles esperados en todo el torneo = share * goles esperados por
    partido * partidos esperados (estos ultimos salen de la simulacion Monte Carlo).

Es matematica pura. Confianza baja: depende de que los goleadores recientes
aproximen bien al plantel y a su reparto de goles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def anytime_scorer_probability(player_lambda):
    """P(el jugador marque al menos un gol) dado su lambda (goles esperados)."""
    return 1.0 - np.exp(-np.asarray(player_lambda, dtype=float))


def match_scorers(team_home, lambda_home, shares_home, team_away, lambda_away, shares_away,
                  *, top_n: int = 10) -> pd.DataFrame:
    """Probabilidad de marcar de cada jugador en un partido concreto.

    shares_home / shares_away : DataFrame con columnas scorer y share del equipo.
    Devuelve los top_n jugadores por probabilidad de marcar en cualquier momento.
    """
    rows = []
    for team, team_lambda, shares in [(team_home, lambda_home, shares_home), (team_away, lambda_away, shares_away)]:
        for r in shares.itertuples(index=False):
            rows.append({
                "player": r.scorer, "team": team,
                "anytime": float(anytime_scorer_probability(team_lambda * r.share)),
            })
    table = pd.DataFrame(rows).sort_values("anytime", ascending=False).reset_index(drop=True)
    return table.head(top_n)


def golden_boot_projection(shares: pd.DataFrame, team_lambda: dict, expected_matches: dict) -> pd.DataFrame:
    """Goles esperados en el torneo por jugador (proyeccion para la Bota de Oro).

    expected_goals = share * goles esperados por partido del equipo * partidos esperados.
    """
    df = shares.copy().rename(columns={"scorer": "player"})
    df["team_lambda"] = df["team"].map(team_lambda)
    df["expected_matches"] = df["team"].map(expected_matches)
    df = df.dropna(subset=["team_lambda", "expected_matches"])
    df["expected_goals"] = df["share"] * df["team_lambda"] * df["expected_matches"]
    return df.sort_values("expected_goals", ascending=False).reset_index(drop=True)


def golden_boot_probabilities(expected_goals, *, n_sims: int = 20_000, seed: int = 0) -> np.ndarray:
    """P(cada jugador sea el maximo goleador del torneo) por simulacion Poisson.

    Muestrea el total de goles de cada jugador como Poisson(expected_goals) y
    cuenta cuantas veces es el mayor. Los empates se rompen por orden (aproximacion).
    """
    expected = np.asarray(expected_goals, dtype=float)
    if expected.size == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    draws = rng.poisson(expected[:, None], size=(expected.size, n_sims))
    winners = draws.argmax(axis=0)
    counts = np.bincount(winners, minlength=expected.size)
    return counts / n_sims
