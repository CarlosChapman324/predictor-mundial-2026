"""Tests de los submodelos de la Capa 2 (corners, remates, tarjetas). Sin red."""

import pandas as pd
import pytest

from model import extras, props


def test_team_rates_calcula_a_favor_y_en_contra():
    stats = pd.DataFrame([
        {"fixture_id": 1, "team": "A", "corners": 8, "shots": 15, "shots_on_target": 6, "fouls": 10, "yellow_cards": 1, "red_cards": 0},
        {"fixture_id": 1, "team": "B", "corners": 2, "shots": 7, "shots_on_target": 2, "fouls": 14, "yellow_cards": 3, "red_cards": 1},
        {"fixture_id": 2, "team": "A", "corners": 6, "shots": 13, "shots_on_target": 5, "fouls": 12, "yellow_cards": 2, "red_cards": 0},
        {"fixture_id": 2, "team": "C", "corners": 4, "shots": 9, "shots_on_target": 3, "fouls": 11, "yellow_cards": 2, "red_cards": 0},
    ])
    rates = props.team_rates(stats).set_index("team")
    assert rates.loc["A", "matches"] == 2
    assert rates.loc["A", "corners_for"] == pytest.approx(7.0)      # (8 + 6) / 2
    assert rates.loc["A", "corners_against"] == pytest.approx(3.0)  # rivales: (2 + 4) / 2
    assert rates.loc["A", "cards_for"] == pytest.approx(1.5)        # (1 + 2) / 2


def test_expected_count_modelo_de_tasa():
    assert props.expected_count(6, 6, 5) == pytest.approx(7.2)               # 6*6/5
    assert props.expected_count(6, 6, 5, strength_mult=1.2) == pytest.approx(8.64)
    assert props.expected_count(8, 6, 5) > props.expected_count(4, 6, 5)     # A genera mas


def test_strength_multiplier():
    assert props.strength_multiplier(1.5, 1.5) == pytest.approx(1.0)
    assert props.strength_multiplier(3.0, 1.5) > 1.0   # equipo dominante
    assert props.strength_multiplier(0.5, 1.5) < 1.0


def test_expected_cards_el_arbitro_pesa_mas():
    flojo = props.expected_cards(2.0, 2.0, referee_cards_per_match=3.0, league_referee_avg=4.0)
    severo = props.expected_cards(2.0, 2.0, referee_cards_per_match=8.0, league_referee_avg=4.0)
    assert severo > flojo  # un arbitro de gatillo facil sube la expectativa
    # Sin arbitro designado (caso 2026 en free): usa el promedio general.
    fallback = props.expected_cards(2.0, 2.0, referee_cards_per_match=None, league_referee_avg=5.0)
    assert fallback == pytest.approx(0.65 * 5.0 + 0.35 * 4.0)


def test_props_se_traducen_a_over_under():
    expected_corners = props.expected_count(6, 6, 5)  # 7.2
    ou = extras.over_under_count(expected_corners, 9.5)
    assert ou["over"] + ou["under"] == pytest.approx(1.0)
    assert ou["under"] > ou["over"]  # con 7.2 esperados, el under 9.5 manda
