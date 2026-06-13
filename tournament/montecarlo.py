"""Motor Monte Carlo del torneo (Fase 3).

Simula el torneo completo muchas veces (>= 10.000): para cada partido muestrea
un marcador desde la matriz del modelo, aplica avance y desempates, y resuelve
el cuadro hasta el campeon. Agrega frecuencias para obtener probabilidad de
clasificar por grupo, de llegar a cada ronda y de ser campeon.

Capa viva (Fase 7): los partidos ya jugados se fijan con su resultado real y
solo se re-simulan los pendientes.

Pendiente de implementar en la Fase 3.
"""

from __future__ import annotations


def simulate_tournament(fixture, model, *, played_results=None, rng=None):
    """Simula un torneo y devuelve el campeon y la ronda alcanzada por equipo."""
    raise NotImplementedError("Fase 3: simulacion de un torneo")


def run_monte_carlo(fixture, model, *, n_sims=10_000, played_results=None, seed=0):
    """Corre n_sims torneos y agrega las probabilidades de avance y campeon."""
    raise NotImplementedError("Fase 3: Monte Carlo")
