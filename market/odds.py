"""Cuotas de mercado y probabilidad implicita (Fase 5).

Convierte cuotas decimales a probabilidad implicita quitando el margen (overround):
    p_i = (1 / cuota_i) / sum_j (1 / cuota_j)
y compara la probabilidad del modelo con la del mercado partido por partido.

Pendiente de implementar en la Fase 5.
"""

from __future__ import annotations


def implied_probabilities(decimal_odds):
    """Probabilidad implicita normalizada (sin margen) de una lista de cuotas."""
    raise NotImplementedError("Fase 5: probabilidad implicita")
