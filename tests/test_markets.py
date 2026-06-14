"""Tests de la derivacion de mercados desde la matriz de marcadores. Sin red."""

import numpy as np
import pytest

from model import goals, markets


def _matrix():
    return goals.score_matrix(1.6, 1.1, rho=-0.1, max_goals=10)


# --- consistencia: cada mercado es una distribucion ------------------------

def test_1x2_suma_1():
    r = markets.one_x_two(_matrix())
    assert sum(r.values()) == pytest.approx(1.0)


def test_over_under_y_btts_suman_1():
    m = _matrix()
    for line in (1.5, 2.5, 3.5):
        ou = markets.over_under(m, line)
        assert ou["over"] + ou["under"] == pytest.approx(1.0)
    btts = markets.both_teams_to_score(m)
    assert btts["yes"] + btts["no"] == pytest.approx(1.0)


def test_doble_oportunidad_es_coherente_con_1x2():
    m = _matrix()
    r = markets.one_x_two(m)
    dc = markets.double_chance(m)
    assert dc["home_or_draw"] == pytest.approx(r["home"] + r["draw"])
    # Las tres dobles oportunidades cuentan cada resultado dos veces -> suman 2.
    assert sum(dc.values()) == pytest.approx(2.0)


# --- direccionalidad y casos conocidos -------------------------------------

def test_one_x_two_en_matriz_hecha_a_mano():
    # 3x3 con valores explicitos: local gana bajo la diagonal, empate en la diagonal.
    m = np.array([
        [0.10, 0.10, 0.05],  # local 0 goles
        [0.20, 0.15, 0.05],  # local 1 gol
        [0.20, 0.05, 0.05],  # local 2 goles
    ])
    r = markets.one_x_two(m)
    assert r["home"] == pytest.approx(0.20 + 0.20 + 0.05)  # (1,0),(2,0),(2,1)
    assert r["draw"] == pytest.approx(0.10 + 0.15 + 0.05)  # diagonal
    assert r["away"] == pytest.approx(0.10 + 0.05 + 0.05)  # (0,1),(0,2),(1,2)


def test_over_es_monotono_en_la_linea():
    m = _matrix()
    assert markets.over_under(m, 1.5)["over"] >= markets.over_under(m, 2.5)["over"]
    assert markets.over_under(m, 2.5)["over"] >= markets.over_under(m, 3.5)["over"]


def test_clean_sheet_y_marcador_exacto():
    m = _matrix()
    cs = markets.clean_sheet(m)
    assert 0.0 <= cs["home"] <= 1.0 and 0.0 <= cs["away"] <= 1.0

    top = markets.exact_score(m, top_n=5)
    assert len(top) == 5
    probs = [s["prob"] for s in top]
    assert probs == sorted(probs, reverse=True)  # ordenados de mayor a menor
    # El marcador mas probable de un 1.6-1.1 con pocos goles debe ser bajo.
    assert top[0]["home_goals"] <= 2 and top[0]["away_goals"] <= 2


def test_all_markets_trae_todo():
    bundle = markets.all_markets(_matrix())
    assert set(bundle) == {"result", "double_chance", "over_under", "btts", "clean_sheet", "exact_score"}
    assert set(bundle["over_under"]) == {"1.5", "2.5", "3.5"}
