"""Tests del modulo de valor: EV, Kelly y backtest de la estrategia. Sin red."""

import numpy as np
import pandas as pd
import pytest

from value import backtest as value_backtest
from value import ev as value_ev


# --- recuperacion de cuota y EV --------------------------------------------

def test_decimal_odds_desde_implicita():
    assert value_ev.decimal_odds_from_implied(0.5, 0.05) == pytest.approx(1 / (0.5 * 1.05))


def test_round_trip_cuota_implicita_cuota():
    odds = np.array([1.9, 3.6, 4.0])
    inverse = 1.0 / odds
    total = inverse.sum()
    implied = inverse / total
    recovered = value_ev.decimal_odds_from_implied(implied, total - 1.0)
    assert np.allclose(recovered, odds)


def test_expected_value():
    assert value_ev.expected_value(0.6, 2.0) == pytest.approx(0.2)
    assert value_ev.expected_value(0.4, 2.0) == pytest.approx(-0.2)


def test_kelly_fraction():
    assert value_ev.kelly_fraction(0.6, 2.0, fraction=1.0) == pytest.approx(0.2)
    assert value_ev.kelly_fraction(0.6, 2.0, fraction=0.5) == pytest.approx(0.1)
    assert value_ev.kelly_fraction(0.5, 2.0) == 0.0   # cuota justa, sin edge
    assert value_ev.kelly_fraction(0.3, 2.0) == 0.0   # edge negativo -> no apostar


def test_annotate_value():
    df = pd.DataFrame([{
        "home_team": "A", "away_team": "B",
        "model_home": 0.6, "model_draw": 0.25, "model_away": 0.15,
        "market_home": 0.5, "market_draw": 0.30, "market_away": 0.20, "overround": 0.05,
    }])
    out = value_ev.annotate_value(df)
    assert out["odds_home"].iloc[0] == pytest.approx(1 / (0.5 * 1.05))
    assert out["ev_home"].iloc[0] == pytest.approx(0.6 / (0.5 * 1.05) - 1)
    assert out["best_bet"].iloc[0] == "home"
    assert bool(out["has_value"].iloc[0]) is True
    assert out["confidence"].iloc[0] == "media"  # prob 0.6, EV moderado: fiable


def test_confianza_baja_en_longshots():
    # El modelo da 12% al local y el mercado ~2%: EV enorme, pero es un longshot
    # donde el modelo esta mal calibrado. Debe marcarse confianza baja.
    df = pd.DataFrame([{
        "home_team": "X", "away_team": "Y",
        "model_home": 0.12, "model_draw": 0.20, "model_away": 0.68,
        "market_home": 0.02, "market_draw": 0.10, "market_away": 0.88, "overround": 0.05,
    }])
    out = value_ev.annotate_value(df)
    assert out["best_bet"].iloc[0] == "home"
    assert out["best_ev"].iloc[0] > 0.5
    assert out["confidence"].iloc[0] == "baja"


# --- backtest de la estrategia ---------------------------------------------

def test_backtest_estrategia_de_valor():
    df = pd.DataFrame([
        {"ev_home": 0.2, "ev_draw": -0.3, "ev_away": -0.3, "odds_home": 2.0, "odds_draw": 4.0, "odds_away": 4.0, "outcome": 0},
        {"ev_home": 0.2, "ev_draw": -0.3, "ev_away": -0.3, "odds_home": 2.0, "odds_draw": 4.0, "odds_away": 4.0, "outcome": 0},
        {"ev_home": -0.1, "ev_draw": -0.1, "ev_away": -0.1, "odds_home": 2.0, "odds_draw": 4.0, "odds_away": 4.0, "outcome": 2},
    ])
    result = value_backtest.backtest_strategy(df)
    value = result["value"]
    assert value["n_bets"] == 2          # solo los dos con EV positivo
    assert value["roi"] == pytest.approx(1.0)  # ambas ganan a cuota 2.0
    assert len(value["bankroll"]) == len(df) + 1
    assert result["favorite"]["n_bets"] == 3   # el favorito siempre apuesta
    assert result["n_settled"] == 3


def test_backtest_sin_partidos_liquidados():
    df = pd.DataFrame([
        {"ev_home": 0.2, "ev_draw": -0.3, "ev_away": -0.3,
         "odds_home": 2.0, "odds_draw": 4.0, "odds_away": 4.0, "outcome": None},
    ])
    result = value_backtest.backtest_strategy(df)
    assert result["n_settled"] == 0
    assert result["value"]["n_bets"] == 0
