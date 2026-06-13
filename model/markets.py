"""Derivacion de los mercados de la Capa 1 desde la matriz de marcadores (Fase 2).

Todos los mercados salen de sumar celdas de la misma matriz que produce
model/goals.score_matrix: 1X2, doble oportunidad, over/under, BTTS, clean sheet
y marcador exacto. Asi quedan internamente consistentes por construccion.

Pendiente de implementar en la Fase 2.
"""

from __future__ import annotations


def one_x_two(matrix):
    """Probabilidad de (gana local, empate, gana visitante)."""
    raise NotImplementedError("Fase 2: mercados")


def over_under(matrix, line):
    """Probabilidad de superar una linea de goles totales (1.5, 2.5, 3.5)."""
    raise NotImplementedError("Fase 2: mercados")


def both_teams_to_score(matrix):
    """Probabilidad de que ambos equipos marquen (BTTS si / no)."""
    raise NotImplementedError("Fase 2: mercados")
