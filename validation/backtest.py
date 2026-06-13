"""Backtesting contra torneos pasados (Fase 4).

Para cada torneo de prueba (Mundiales 2014/2018/2022, Eurocopas recientes) se
entrena el modelo SOLO con datos anteriores a ese torneo (corte temporal
estricto, sin mirar el futuro) y se predicen sus partidos. Se reportan las
metricas frente a baselines: uniforme, ranking FIFA y, si se consigue, mercado.

Pendiente de implementar en la Fase 4.
"""

from __future__ import annotations


def backtest_tournament(matches, tournament_name, cutoff_date):
    """Entrena con datos < cutoff_date y evalua las predicciones del torneo."""
    raise NotImplementedError("Fase 4: backtesting")
