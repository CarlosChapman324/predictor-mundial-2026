"""Modelo de goles: Poisson bivariado con correccion de Dixon-Coles (Fase 2).

A partir de las fuerzas de ataque/defensa de cada seleccion (estimadas por
maxima verosimilitud sobre el historico, con ponderacion temporal) y la ventaja
de localia, se obtienen lambda_local y lambda_visitante y de ahi la matriz de
probabilidad de cada marcador i-j. Esa matriz es la fuente unica de todos los
mercados de la Capa 1 (ver model/markets.py).

Pendiente de implementar en la Fase 2.
"""

from __future__ import annotations


def fit_goal_model(matches, *, half_life_days, host_teams):
    """Estima ataque/defensa por seleccion, localia y el parametro de
    dependencia de Dixon-Coles por maxima verosimilitud ponderada."""
    raise NotImplementedError("Fase 2: modelo de goles")


def score_matrix(lambda_home, lambda_away, *, rho, max_goals=10):
    """Matriz (max_goals+1 x max_goals+1) de probabilidad de cada marcador,
    con el ajuste de Dixon-Coles en los marcadores bajos."""
    raise NotImplementedError("Fase 2: matriz de marcadores")
