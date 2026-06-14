"""Metricas de evaluacion probabilistica para predicciones 1X2.

Convencion: una prediccion es un vector ordenado [P(local), P(empate), P(visitante)]
y el resultado observado es un indice 0 (gana local), 1 (empate) o 2 (gana visitante).

Por que estas metricas:
  - RPS (Ranked Probability Score): la mas apropiada para el 1X2 porque respeta
    el ORDEN de los resultados (equivocarse prediciendo visitante cuando gano el
    local penaliza mas que predecir empate). Es el estandar en pronostico de futbol.
  - log-loss: penaliza con dureza la confianza mal puesta (predecir 1% a lo que pasa).
  - Brier: error cuadratico multiclase, intuitivo y acotado.
  - Calibracion: cuando el modelo dice 60%, .pasa el 60% de las veces? Se mide
    agrupando las predicciones por probabilidad (reliability diagram).
Menor es mejor en RPS, log-loss y Brier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ranked_probability_score(probs, outcome: int) -> float:
    """RPS de una prediccion ordinal frente al resultado observado.

    RPS = (1 / (r-1)) * sum_i (CumPred_i - CumObs_i)^2, sobre las r-1 primeras
    categorias (la ultima acumulada siempre es 1 y no aporta).
    """
    probs = np.asarray(probs, dtype=float)
    r = len(probs)
    observed = np.zeros(r)
    observed[outcome] = 1.0
    cum_pred = np.cumsum(probs)
    cum_obs = np.cumsum(observed)
    return float(np.sum((cum_pred[:-1] - cum_obs[:-1]) ** 2) / (r - 1))


def log_loss(probs, outcome: int, eps: float = 1e-15) -> float:
    """Log-loss (entropia cruzada): -log de la probabilidad asignada al resultado."""
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    return float(-np.log(probs[outcome]))


def brier_score(probs, outcome: int) -> float:
    """Brier multiclase: suma de (prob - indicador)^2 sobre las categorias."""
    probs = np.asarray(probs, dtype=float)
    observed = np.zeros(len(probs))
    observed[outcome] = 1.0
    return float(np.sum((probs - observed) ** 2))


def summarize(predictions, outcomes) -> dict:
    """Promedia las tres metricas sobre un conjunto de predicciones.

    predictions : array (n, 3). outcomes : array (n,) de indices 0/1/2.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n = len(outcomes)
    rps = np.mean([ranked_probability_score(predictions[i], outcomes[i]) for i in range(n)])
    ll = np.mean([log_loss(predictions[i], outcomes[i]) for i in range(n)])
    brier = np.mean([brier_score(predictions[i], outcomes[i]) for i in range(n)])
    return {"n": int(n), "rps": float(rps), "log_loss": float(ll), "brier": float(brier)}


def calibration_table(predictions, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de calibracion (reliability diagram) agrupando todas las
    predicciones por categoria en bins de probabilidad.

    Cada par (partido, categoria) es un pronostico binario: .ocurrio esa
    categoria? Se compara la probabilidad media predicha con la frecuencia real.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)

    probs, hits = [], []
    for i in range(len(outcomes)):
        for c in range(predictions.shape[1]):
            probs.append(predictions[i, c])
            hits.append(1.0 if outcomes[i] == c else 0.0)
    probs = np.asarray(probs)
    hits = np.asarray(hits)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "mean_predicted": float(probs[mask].mean()),
            "observed_frequency": float(hits[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)
