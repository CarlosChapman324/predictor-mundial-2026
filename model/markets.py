"""Mercados de la Capa 1 derivados de la matriz de marcadores.

Todos salen de la misma matriz que produce model.goals.score_matrix (filas =
goles del local, columnas = goles del visitante). Como cada mercado es una suma
de celdas de esa unica matriz, quedan internamente consistentes por construccion
(por ejemplo, 1X2 siempre suma 1 y el over + under tambien).

Convencion: la matriz ya viene normalizada (suma 1), asi que cada salida es una
probabilidad directa.
"""

from __future__ import annotations

import numpy as np


def _indices(matrix: np.ndarray):
    """Rejillas de goles de local (filas) y visitante (columnas)."""
    n = matrix.shape[0]
    home_goals = np.arange(n)[:, None]   # varia por fila
    away_goals = np.arange(n)[None, :]   # varia por columna
    return home_goals, away_goals


def one_x_two(matrix: np.ndarray) -> dict[str, float]:
    """Resultado 1X2: gana local, empate, gana visitante.

    Local gana en la parte de abajo de la diagonal (mas goles del local),
    empate en la diagonal, visitante gana arriba.
    """
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    return {"home": home_win, "draw": draw, "away": away_win}


def double_chance(matrix: np.ndarray) -> dict[str, float]:
    """Doble oportunidad: 1X (local o empate), 12 (no empate), X2 (empate o visitante)."""
    r = one_x_two(matrix)
    return {
        "home_or_draw": r["home"] + r["draw"],
        "home_or_away": r["home"] + r["away"],
        "draw_or_away": r["draw"] + r["away"],
    }


def over_under(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Over/Under de goles totales para una linea (1.5, 2.5, 3.5...).

    La linea es semientera, asi que no hay empates: over = total > line.
    """
    home_goals, away_goals = _indices(matrix)
    total = home_goals + away_goals
    over = float(matrix[total > line].sum())
    return {"over": over, "under": 1.0 - over}


def both_teams_to_score(matrix: np.ndarray) -> dict[str, float]:
    """BTTS: probabilidad de que ambos marquen (ambos >= 1 gol)."""
    yes = float(matrix[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def clean_sheet(matrix: np.ndarray) -> dict[str, float]:
    """Porteria a cero: el local no recibe (columna 0) / el visitante no recibe (fila 0)."""
    home_clean_sheet = float(matrix[:, 0].sum())  # visitante marca 0
    away_clean_sheet = float(matrix[0, :].sum())  # local marca 0
    return {"home": home_clean_sheet, "away": away_clean_sheet}


def exact_score(matrix: np.ndarray, top_n: int = 5) -> list[dict]:
    """Los top_n marcadores mas probables, de mayor a menor probabilidad."""
    flat = matrix.ravel()
    order = np.argsort(flat)[::-1][:top_n]
    n = matrix.shape[1]
    results = []
    for idx in order:
        home, away = divmod(int(idx), n)
        results.append({"home_goals": home, "away_goals": away, "prob": float(flat[idx])})
    return results


def all_markets(matrix: np.ndarray, *, ou_lines=(1.5, 2.5, 3.5), top_scores: int = 5) -> dict:
    """Conveniencia: todos los mercados de la Capa 1 de un partido en un dict.

    Pensado para que la capa de presentacion (Streamlit) consuma una sola
    estructura por partido.
    """
    return {
        "result": one_x_two(matrix),
        "double_chance": double_chance(matrix),
        "over_under": {f"{line}": over_under(matrix, line) for line in ou_lines},
        "btts": both_teams_to_score(matrix),
        "clean_sheet": clean_sheet(matrix),
        "exact_score": exact_score(matrix, top_scores),
    }
