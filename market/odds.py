"""Cuotas de mercado, probabilidad implicita y comparacion modelo vs mercado.

Angulo financiero del proyecto. Una casa de apuestas publica cuotas decimales;
su inversa 1/cuota es una probabilidad implicita, pero las inversas suman MAS de
1: ese exceso es el margen de la casa (overround). Quitarlo da la probabilidad
implicita "limpia" del mercado:

    p_i = (1/cuota_i) / sum_j (1/cuota_j)

Luego se compara la probabilidad del modelo con la del mercado partido por
partido. Marco honesto: el mercado suele ser MUY eficiente, asi que una
discrepancia grande casi siempre delata un error o un dato que le falta al
modelo, no una oportunidad real. Es un estudio de eficiencia de mercado, no un
sistema de apuestas.

Logica pura y testeable; la ingesta de cuotas vive en market/ingest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from validation import metrics

OUTCOMES = ("home", "draw", "away")


def implied_probabilities(decimal_odds) -> np.ndarray:
    """Probabilidad implicita normalizada (sin margen) de una lista de cuotas.

    Metodo proporcional: se reparte el overround en proporcion a cada inversa.
    Es el estandar y el mas transparente; existen alternativas (Shin, metodo de
    potencias) que modelan el margen de forma no uniforme, pero aqui se prefiere
    la claridad.
    """
    odds = np.asarray(decimal_odds, dtype=float)
    inverse = 1.0 / odds
    return inverse / inverse.sum()


def overround(decimal_odds) -> float:
    """Margen de la casa: cuanto suman las inversas por encima de 1 (p. ej. 0.05 = 5%)."""
    odds = np.asarray(decimal_odds, dtype=float)
    return float(np.sum(1.0 / odds) - 1.0)


def add_market_probabilities(
    df: pd.DataFrame, *, home="home_odds", draw="draw_odds", away="away_odds"
) -> pd.DataFrame:
    """Anade columnas market_home/draw/away (implicitas sin margen) y overround."""
    out = df.copy()
    inv = pd.DataFrame({
        "home": 1.0 / df[home],
        "draw": 1.0 / df[draw],
        "away": 1.0 / df[away],
    })
    total = inv.sum(axis=1)
    out["market_home"] = inv["home"] / total
    out["market_draw"] = inv["draw"] / total
    out["market_away"] = inv["away"] / total
    out["overround"] = total - 1.0
    return out


def add_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Anade el 'edge' del modelo (modelo - mercado) por resultado.

    Requiere columnas model_* y market_*. edge positivo = el modelo da mas
    probabilidad que el mercado a ese resultado (donde "veria valor").
    """
    out = df.copy()
    for o in OUTCOMES:
        out[f"edge_{o}"] = df[f"model_{o}"] - df[f"market_{o}"]
    out["max_abs_edge"] = out[[f"edge_{o}" for o in OUTCOMES]].abs().max(axis=1)
    model_fav = df[[f"model_{o}" for o in OUTCOMES]].to_numpy().argmax(axis=1)
    market_fav = df[[f"market_{o}" for o in OUTCOMES]].to_numpy().argmax(axis=1)
    out["model_favorite"] = [OUTCOMES[i] for i in model_fav]
    out["market_favorite"] = [OUTCOMES[i] for i in market_fav]
    out["same_favorite"] = model_fav == market_fav
    return out


def efficiency_summary(df: pd.DataFrame, *, outcome_col: str | None = None) -> dict:
    """Resumen de eficiencia de mercado sobre un conjunto de partidos comparados.

    Si se pasa outcome_col (0/1/2 del resultado real) sobre partidos ya jugados,
    tambien compara quien predijo mejor (RPS del modelo vs RPS del mercado).
    """
    summary = {
        "n": int(len(df)),
        "avg_overround": float(df["overround"].mean()),
        "agreement_favorite": float(df["same_favorite"].mean()),
        "mean_abs_edge": float(df["max_abs_edge"].mean()),
    }
    model = df[[f"model_{o}" for o in OUTCOMES]].to_numpy()
    market = df[[f"market_{o}" for o in OUTCOMES]].to_numpy()
    summary["prob_correlation"] = float(np.corrcoef(model.ravel(), market.ravel())[0, 1])

    if outcome_col is not None and outcome_col in df.columns:
        settled = df[df[outcome_col].notna()]
        if not settled.empty:
            y = settled[outcome_col].to_numpy(dtype=int)
            m = settled[[f"model_{o}" for o in OUTCOMES]].to_numpy()
            k = settled[[f"market_{o}" for o in OUTCOMES]].to_numpy()
            summary["n_settled"] = int(len(settled))
            summary["rps_model"] = metrics.summarize(m, y)["rps"]
            summary["rps_market"] = metrics.summarize(k, y)["rps"]
    return summary
