"""Tests de la Capa 2: goleadores, Bota de Oro, tarjetas y corners. Sin red."""

import numpy as np
import pandas as pd
import pytest

from data import scorers as scorers_data
from model import extras, scorers


def test_anytime_scorer_probability():
    assert scorers.anytime_scorer_probability(0.0) == pytest.approx(0.0)
    assert scorers.anytime_scorer_probability(0.5) == pytest.approx(1 - np.exp(-0.5))
    assert scorers.anytime_scorer_probability(10.0) > 0.99


def test_player_shares_suma_1_y_excluye_autogoles():
    g = pd.DataFrame([
        {"date": "2024-01-01", "team": "A", "scorer": "X", "own_goal": False},
        {"date": "2024-02-01", "team": "A", "scorer": "X", "own_goal": False},
        {"date": "2024-03-01", "team": "A", "scorer": "Y", "own_goal": False},
        {"date": "2024-04-01", "team": "A", "scorer": "Z", "own_goal": True},   # autogol: fuera
        {"date": "2024-01-01", "team": "B", "scorer": "W", "own_goal": False},
    ])
    shares = scorers_data.player_shares(g)
    a = shares[shares["team"] == "A"]
    assert a["share"].sum() == pytest.approx(1.0)
    assert "Z" not in set(a["scorer"])
    assert a[a["scorer"] == "X"]["share"].iloc[0] == pytest.approx(2 / 3)


def test_match_scorers_ordena_y_limita():
    sh_a = pd.DataFrame({"scorer": ["X", "Y"], "share": [0.7, 0.3]})
    sh_b = pd.DataFrame({"scorer": ["W"], "share": [1.0]})
    table = scorers.match_scorers("A", 2.0, sh_a, "B", 0.5, sh_b, top_n=2)
    assert len(table) == 2
    assert list(table["anytime"]) == sorted(table["anytime"], reverse=True)
    assert table.iloc[0]["player"] == "X"  # mayor lambda (2.0 * 0.7)


def test_golden_boot_proyeccion_y_probabilidad():
    shares = pd.DataFrame({
        "team": ["A", "A", "B"], "scorer": ["X", "Y", "W"],
        "goals": [10, 5, 8], "share": [2 / 3, 1 / 3, 1.0],
    })
    gb = scorers.golden_boot_projection(shares, {"A": 1.8, "B": 1.0}, {"A": 6, "B": 4})
    assert {"player", "team", "expected_goals"} <= set(gb.columns)
    assert gb[gb["player"] == "X"]["expected_goals"].iloc[0] == pytest.approx(2 / 3 * 1.8 * 6)
    assert gb.iloc[0]["player"] == "X"  # el de mayor goles esperados

    probs = scorers.golden_boot_probabilities(gb["expected_goals"].to_numpy(), n_sims=5000, seed=1)
    assert probs.sum() == pytest.approx(1.0)        # un ganador por simulacion
    assert probs[0] == probs.max()                  # el favorito gana mas seguido


def test_tarjetas_y_corners():
    ou = extras.over_under_count(4.5, 4.5)
    assert ou["over"] + ou["under"] == pytest.approx(1.0)
    # Linea mas baja -> mas probabilidad de over.
    assert extras.over_under_count(4.5, 3.5)["over"] > extras.over_under_count(4.5, 5.5)["over"]
    assert 0.0 < extras.red_card_probability(0.2) < 1.0
    assert extras.red_card_probability(1.0) > extras.red_card_probability(0.2)
