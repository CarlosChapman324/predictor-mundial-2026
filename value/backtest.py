"""Backtest de la estrategia de valor: ROI, yield y bankroll frente a baselines.

Regla: apostar 1 unidad a la seleccion de EV positivo de cada partido y
liquidarla con el resultado real. Se compara contra dos baselines: apostar
siempre al favorito (cuota mas baja) y apostar al azar.

Honestidad (clave del proyecto): lo mas probable es que el edge sea marginal o
NEGATIVO tras el margen de la casa. Reportarlo con franqueza es parte del valor,
no un fallo. Para un backtest sobre torneos pasados harian falta cuotas
historicas de selecciones (no gratuitas); este motor corre sobre cualquier set
de partidos con cuotas y resultado (incluido el propio Mundial a medida que avanza).

Matematica pura, sin red.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from value.ev import OUTCOMES


def _run(picks, odds, outcomes, stake):
    """Liquida una lista de selecciones (idx por partido, o -1 para no apostar)."""
    profit, staked, n = 0.0, 0.0, 0
    bankroll = [0.0]
    for i, pick in enumerate(picks):
        if pick < 0:
            bankroll.append(profit)
            continue
        n += 1
        staked += stake
        profit += stake * (odds[i, pick] - 1.0) if pick == outcomes[i] else -stake
        bankroll.append(profit)
    roi = profit / staked if staked else 0.0
    return {"n_bets": n, "profit": float(profit), "staked": float(staked),
            "roi": float(roi), "bankroll": bankroll}


def backtest_strategy(df: pd.DataFrame, *, outcome_col: str = "outcome",
                      ev_threshold: float = 0.0, stake: float = 1.0, seed: int = 0) -> dict:
    """Backtest de apostar a las selecciones con EV > ev_threshold.

    df necesita ev_{home,draw,away}, odds_{home,draw,away} y outcome_col (0/1/2).
    Devuelve metricas de la estrategia de valor y de los baselines favorito y azar.
    """
    settled = df[df[outcome_col].notna()].copy()
    if settled.empty:
        empty = {"n_bets": 0, "profit": 0.0, "staked": 0.0, "roi": 0.0, "bankroll": [0.0]}
        return {"value": empty, "favorite": empty, "random": empty, "n_settled": 0}

    ev = settled[[f"ev_{o}" for o in OUTCOMES]].to_numpy()
    odds = settled[[f"odds_{o}" for o in OUTCOMES]].to_numpy()
    outcomes = settled[outcome_col].to_numpy(dtype=int)

    value_picks = np.where(ev.max(axis=1) > ev_threshold, ev.argmax(axis=1), -1)
    favorite_picks = odds.argmin(axis=1)  # cuota mas baja = favorito del mercado
    random_picks = np.random.default_rng(seed).integers(0, len(OUTCOMES), len(settled))

    return {
        "value": _run(value_picks, odds, outcomes, stake),
        "favorite": _run(favorite_picks, odds, outcomes, stake),
        "random": _run(random_picks, odds, outcomes, stake),
        "n_settled": int(len(settled)),
    }
