"""Submodelos de la Capa 2: corners, remates y tarjetas (experimental, confianza baja).

Son modelos de CONTEO. La cantidad esperada de cada evento en un partido se modela
como un Poisson y de ahi salen los over/under (ver model/extras). Este modulo
estima la TASA esperada de cada evento:

  - Corners y remates: tasa "a favor" del equipo por tasa "en contra" del rival,
    normalizada por la media (modelo ataque x defensa, igual que el de goles),
    ajustada por la fuerza relativa (un equipo dominante genera algo mas).
  - Tarjetas: mezcla de la propension de los equipos y el promedio del arbitro
    designado, que PESA MAS que los equipos. Sin arbitro (caso 2026 en plan free)
    se usa el promedio general como fallback.

Las tasas vienen de los datos de API-Football (Fase 1). Es matematica pura, sin
red. Confianza baja: muestra historica limitada y mercados ruidosos; nunca se
presentan con la autoridad del resultado o los goles.
"""

from __future__ import annotations

import pandas as pd

CARD_REFEREE_WEIGHT = 0.65  # el arbitro pesa mas que los equipos en las tarjetas
STRENGTH_EXPONENT = 0.5     # cuanto influye la dominancia (del modelo de goles)


def team_rates(team_match_stats: pd.DataFrame) -> pd.DataFrame:
    """Tasas por equipo: eventos a favor y en contra por partido.

    Para cada equipo promedia sus propias stats (a favor) y las del rival del
    mismo partido (en contra). Una fila por equipo.
    """
    df = team_match_stats
    opponent = df[["fixture_id", "team", "corners", "shots", "shots_on_target"]].rename(
        columns={"team": "opp", "corners": "corners_opp",
                 "shots": "shots_opp", "shots_on_target": "sot_opp"}
    )
    merged = df.merge(opponent, on="fixture_id")
    merged = merged[merged["team"] != merged["opp"]].copy()
    merged["cards"] = merged["yellow_cards"].fillna(0) + merged["red_cards"].fillna(0)

    g = merged.groupby("team")
    return pd.DataFrame({
        "matches": g.size(),
        "corners_for": g["corners"].mean(),
        "corners_against": g["corners_opp"].mean(),
        "shots_for": g["shots"].mean(),
        "shots_against": g["shots_opp"].mean(),
        "sot_for": g["shots_on_target"].mean(),
        "fouls_for": g["fouls"].mean(),
        "cards_for": g["cards"].mean(),
        "reds_for": g["red_cards"].apply(lambda s: s.fillna(0).mean()),
    }).reset_index()


def expected_count(for_rate_a: float, against_rate_b: float, league_avg: float,
                   strength_mult: float = 1.0) -> float:
    """Tasa esperada de un evento (corners, remates) para el equipo A.

    Modelo de tasa ataque x defensa: lo que A genera por lo que B concede,
    normalizado por la media de la liga, y escalado por la dominancia de A.
    """
    if league_avg <= 0:
        return float(for_rate_a) * strength_mult
    return float(for_rate_a) * float(against_rate_b) / float(league_avg) * strength_mult


def strength_multiplier(lambda_team: float, lambda_average: float,
                        exponent: float = STRENGTH_EXPONENT) -> float:
    """Factor de dominancia desde el modelo de goles: un equipo con mas goles
    esperados que la media genera algo mas de corners y remates."""
    if lambda_average <= 0:
        return 1.0
    return (lambda_team / lambda_average) ** exponent


def expected_cards(cards_rate_a: float, cards_rate_b: float,
                   referee_cards_per_match: float | None, league_referee_avg: float,
                   *, weight: float = CARD_REFEREE_WEIGHT) -> float:
    """Tarjetas totales esperadas en el partido.

    Mezcla la propension de los equipos (tarjetas que suelen recibir) con el
    promedio del arbitro, que pesa mas (weight). Si no hay arbitro designado
    (caso 2026 en el plan free), usa el promedio general de arbitros.
    """
    team_cards = float(cards_rate_a) + float(cards_rate_b)
    referee = league_referee_avg if referee_cards_per_match is None else referee_cards_per_match
    return weight * float(referee) + (1.0 - weight) * team_cards
