"""Elo propio calculado desde el historico de partidos.

Por que un Elo propio y no el ranking FIFA: el ranking FIFA es opaco y los
clones virales lo usan tal cual. Calcular el Elo desde el registro de partidos
es transparente, se puede explicar paso a paso y se puede validar contra
eloratings.net. Es uno de los diferenciadores del proyecto.

Metodo: 'World Football Elo Ratings', el estandar para selecciones. Para cada
partido (en orden cronologico) se actualiza el rating de los dos equipos:

    R_nuevo = R_viejo + K * G * (W - We)

  - W   = resultado real para el local (1 gana, 0.5 empata, 0 pierde)
  - We  = resultado esperado segun la diferencia de rating (curva logistica)
  - K   = importancia del partido (un Mundial pesa el triple que un amistoso)
  - G   = multiplicador por diferencia de goles (ganar 4-0 mueve mas que 1-0)

Es matematica pura: no toca la red. Recibe el DataFrame ya normalizado y
devuelve el historico de ratings y el rating actual de cada seleccion.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

# Rating inicial de toda seleccion que aparece por primera vez.
BASE_RATING = 1500.0

# Ventaja de localia, en puntos de rating. Solo se aplica si el partido NO es en
# cancha neutral. En el Mundial 2026 esto solo beneficia a los anfitriones
# (Mexico, Canada, USA) cuando juegan de locales; el resto juega en neutral.
HOME_ADVANTAGE = 100.0

# Factor K por categoria de torneo (esquema clasico de World Football Elo).
# Cuanto mas importante el partido, mas se ajusta el rating tras el resultado.
K_BY_CATEGORY = {
    "world_cup": 60.0,
    "confederations": 50.0,
    "continental": 50.0,
    "qualifier": 40.0,
    "nations_league": 40.0,
    "other": 30.0,
    "friendly": 20.0,
}
DEFAULT_K = 30.0


def expected_score(rating_diff: float) -> float:
    """Probabilidad esperada de ganar del local segun la diferencia de rating.

    rating_diff ya debe incluir la ventaja de localia. Es la curva logistica
    estandar del Elo: +400 de ventaja equivale a ~10 veces mas probable.
    """
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def goal_difference_index(goal_diff: int) -> float:
    """Multiplicador por margen de victoria (formula de World Football Elo).

    Empate o victoria por 1 -> 1.0; por 2 -> 1.5; de 3 en adelante crece mas
    despacio para que las goleadas no disparen el rating de forma absurda.
    """
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def k_factor(category: str, k_by_category: dict[str, float] | None = None) -> float:
    """Factor K segun la categoria del torneo."""
    table = k_by_category or K_BY_CATEGORY
    return table.get(category, DEFAULT_K)


def _match_result(home_score: int, away_score: int) -> float:
    """Resultado real para el local: 1.0 gana, 0.5 empata, 0.0 pierde."""
    if home_score > away_score:
        return 1.0
    if home_score == away_score:
        return 0.5
    return 0.0


def compute_elo(
    matches: pd.DataFrame,
    *,
    base_rating: float = BASE_RATING,
    home_advantage: float = HOME_ADVANTAGE,
    k_by_category: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula el Elo recorriendo los partidos en orden cronologico.

    Parametros
    ----------
    matches : DataFrame normalizado (ver data/ingest.normalize_results), ordenado
        por fecha, con columnas date, home_team, away_team, home_score,
        away_score, neutral, tournament_category.

    Devuelve
    --------
    history : un registro por equipo y por partido, con su rating DESPUES del
        partido. Sirve para graficar la evolucion y para tomar el rating "a
        fecha X" en el backtesting (sin mirar el futuro).
    current : el rating final de cada seleccion, con partidos jugados y ultima
        fecha. Ordenado de mayor a menor.
    """
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date")

    ratings: dict[str, float] = defaultdict(lambda: base_rating)
    games_played: dict[str, int] = defaultdict(int)
    last_date: dict[str, pd.Timestamp] = {}
    history: list[dict] = []

    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home, r_away = ratings[home], ratings[away]

        advantage = 0.0 if row.neutral else home_advantage
        we_home = expected_score(r_home + advantage - r_away)
        w_home = _match_result(row.home_score, row.away_score)
        g = goal_difference_index(row.home_score - row.away_score)
        k = k_factor(row.tournament_category, k_by_category)

        # El ajuste es de suma cero: lo que gana el local lo pierde el visitante,
        # asi que el Elo total del sistema se conserva (util como test).
        delta = k * g * (w_home - we_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

        for team in (home, away):
            games_played[team] += 1
            last_date[team] = row.date
        history.append({"date": row.date, "team": home, "rating": ratings[home]})
        history.append({"date": row.date, "team": away, "rating": ratings[away]})

    history_df = pd.DataFrame(history, columns=["date", "team", "rating"])

    current_df = (
        pd.DataFrame(
            {
                "team": list(ratings.keys()),
                "rating": list(ratings.values()),
                "games_played": [games_played[t] for t in ratings],
                "last_played": [last_date[t] for t in ratings],
            }
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    return history_df, current_df
