"""Metricas de evaluacion probabilistica (Fase 4).

RPS (Ranked Probability Score, el mas apropiado para el 1X2 por ser ordinal),
log-loss, Brier score y utilidades de calibracion (reliability diagram).

Pendiente de implementar en la Fase 4.
"""

from __future__ import annotations


def ranked_probability_score(probs, outcome):
    """RPS de una prediccion 1X2 frente al resultado observado."""
    raise NotImplementedError("Fase 4: metricas")


def log_loss(probs, outcome):
    """Log-loss (entropia cruzada) de una prediccion frente al resultado."""
    raise NotImplementedError("Fase 4: metricas")
