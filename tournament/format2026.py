"""Formato del Mundial 2026: grupos, desempates y asignacion de mejores terceros.

48 equipos, 12 grupos (A-L) de 4. Avanzan 1o y 2o de cada grupo mas los 8
mejores terceros (32 a la ronda de eliminatorias). Desempates de grupo en el
orden oficial 2026: puntos, luego enfrentamiento directo ANTES que diferencia de
goles global. Los mejores terceros se rankean por criterios globales y se
asignan a las llaves segun la tabla de combinaciones de FIFA.

Pendiente de implementar en la Fase 3.
"""

from __future__ import annotations


def rank_group(group_matches):
    """Ordena los 4 equipos de un grupo aplicando los desempates 2026."""
    raise NotImplementedError("Fase 3: desempates de grupo")


def best_thirds(group_tables):
    """Elige los 8 mejores terceros y los asigna a sus llaves del cuadro."""
    raise NotImplementedError("Fase 3: mejores terceros")
