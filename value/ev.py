"""Valor esperado (EV) y mejor apuesta del partido.

EV = p_modelo * cuota - 1. Positivo significa que el modelo asigna mas
probabilidad de la que implica la cuota, descontado YA el margen de la casa (la
cuota real lo incluye). Por eso un EV positivo exige que el modelo le gane al
mercado por mas que el margen.

Solo aplica a mercados con cuotas (1X2, doble oportunidad, totales); los props no
tienen cuotas gratis y no entran al analisis de valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")


def decimal_odds_from_implied(implied_prob, overround):
    """Recupera la cuota decimal desde la probabilidad implicita (sin margen) y el
    overround. En el metodo proporcional: 1/cuota = implicita * (1 + overround)."""
    implied = np.asarray(implied_prob, dtype=float)
    return 1.0 / (implied * (1.0 + np.asarray(overround, dtype=float)))


def expected_value(model_prob, decimal_odds):
    """EV de apostar 1 unidad a un resultado: p_modelo * cuota - 1."""
    return np.asarray(model_prob, dtype=float) * np.asarray(decimal_odds, dtype=float) - 1.0


def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.5) -> float:
    """Fraccion del bankroll a apostar (Kelly fraccionado). 0 si no hay edge.

    Kelly completo: f = (b*p - q) / b, con b = cuota - 1, q = 1 - p. Se devuelve
    una fraccion de ese f (por defecto la mitad) para reducir la varianza.
    """
    p = float(model_prob)
    b = float(decimal_odds) - 1.0
    if b <= 0:
        return 0.0
    full = (b * p - (1.0 - p)) / b
    return max(0.0, full) * fraction


def annotate_value(comparison: pd.DataFrame) -> pd.DataFrame:
    """Anade cuotas implicitas, EV por resultado y la mejor apuesta del partido.

    comparison debe traer model_{home,draw,away}, market_{home,draw,away} (implicitas
    sin margen) y overround. Devuelve una copia con odds_*, ev_*, best_bet, best_ev
    y has_value (EV positivo).
    """
    df = comparison.copy()
    for outcome in OUTCOMES:
        df[f"odds_{outcome}"] = decimal_odds_from_implied(df[f"market_{outcome}"], df["overround"])
        df[f"ev_{outcome}"] = expected_value(df[f"model_{outcome}"], df[f"odds_{outcome}"])

    ev_matrix = df[[f"ev_{o}" for o in OUTCOMES]].to_numpy()
    best_idx = ev_matrix.argmax(axis=1)
    df["best_bet"] = [OUTCOMES[i] for i in best_idx]
    df["best_ev"] = ev_matrix.max(axis=1)
    df["has_value"] = df["best_ev"] > 0

    # Probabilidad del modelo para la apuesta elegida y nivel de CONFIANZA.
    # El mercado es eficiente: un EV implausiblemente alto (> 50%) o una apuesta a
    # un favorito largo (prob del modelo < 20%) caen donde el modelo esta peor
    # calibrado, asi que casi siempre delatan un dato que le falta, no valor real.
    model_matrix = df[[f"model_{o}" for o in OUTCOMES]].to_numpy()
    df["pick_prob"] = model_matrix[np.arange(len(df)), best_idx] if len(df) else []
    unreliable = (df["pick_prob"] < 0.20) | (df["best_ev"] > 0.50)
    df["confidence"] = np.where(unreliable, "baja", "media")
    return df
