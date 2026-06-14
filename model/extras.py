"""Tarjetas y corners (Capa 2, experimental, confianza baja).

Estos mercados son los mas ruidosos del proyecto. La LOGICA es simple: dado un
numero esperado de tarjetas o corners en el partido, se modela el total como un
Poisson y se derivan los over/under y la probabilidad de tarjeta roja.

Lo que NO es trivial es estimar esa tasa esperada: requiere datos granulares por
partido (tarjetas y corners historicos por seleccion, ajustados por la intensidad
del partido). No hay una fuente gratuita de eso para selecciones; se obtendria de
API-Football (plan free, con cache diario). Por eso aqui van solo las funciones
puras y testeables, y la tasa entra como parametro. Sin datos reales, estos
mercados NO se muestran en el dashboard (no se inventan numeros).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def over_under_count(expected: float, line: float) -> dict[str, float]:
    """Over/Under de un conteo (tarjetas, corners) modelado como Poisson.

    La linea es semientera, asi que over = P(total > line) = 1 - P(total <= floor(line)).
    """
    under = float(poisson.cdf(np.floor(line), expected))
    return {"over": 1.0 - under, "under": under}


def red_card_probability(expected_reds: float) -> float:
    """P(haya al menos una tarjeta roja) dado el numero esperado de rojas."""
    return float(1.0 - np.exp(-expected_reds))
