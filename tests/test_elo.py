"""Tests del Elo. Matematica pura: datos sinteticos, sin red ni Parquet."""

import pandas as pd
import pytest

from model import elo


def _match(date, home, away, hs, as_, neutral=True, category="friendly"):
    return {
        "date": pd.Timestamp(date), "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "neutral": neutral,
        "tournament_category": category,
    }


# --- piezas elementales ----------------------------------------------------

def test_expected_score_simetrico_y_monotono():
    # Sin diferencia de rating, ambos tienen 50%.
    assert elo.expected_score(0.0) == pytest.approx(0.5)
    # Mas rating del local -> mayor probabilidad esperada.
    assert elo.expected_score(200.0) > 0.5
    # +400 de ventaja -> ~10/11 (diez veces mas probable).
    assert elo.expected_score(400.0) == pytest.approx(10 / 11, abs=1e-6)
    # Simetria: lo que gana uno lo pierde el otro.
    assert elo.expected_score(150.0) + elo.expected_score(-150.0) == pytest.approx(1.0)


def test_multiplicador_por_diferencia_de_goles():
    assert elo.goal_difference_index(0) == 1.0
    assert elo.goal_difference_index(1) == 1.0
    assert elo.goal_difference_index(-2) == 1.5   # usa el valor absoluto
    assert elo.goal_difference_index(3) == pytest.approx(1.75)
    assert elo.goal_difference_index(4) == pytest.approx(1.875)


def test_factor_k_por_categoria():
    assert elo.k_factor("world_cup") == 60.0
    assert elo.k_factor("friendly") == 20.0
    assert elo.k_factor("categoria_inexistente") == elo.DEFAULT_K


# --- compute_elo ------------------------------------------------------------

def test_el_elo_total_se_conserva():
    # Cada partido reparte puntos de suma cero, asi que el total no cambia.
    matches = pd.DataFrame([
        _match("2020-01-01", "A", "B", 3, 0),
        _match("2020-02-01", "B", "C", 1, 1),
        _match("2020-03-01", "C", "A", 0, 2),
    ])
    _, current = elo.compute_elo(matches)
    total = current["rating"].sum()
    assert total == pytest.approx(len(current) * elo.BASE_RATING)


def test_el_que_siempre_gana_sube_y_el_que_pierde_baja():
    matches = pd.DataFrame([
        _match("2020-01-01", "Fuerte", "Debil", 3, 0),
        _match("2020-02-01", "Debil", "Fuerte", 0, 2),
        _match("2020-03-01", "Fuerte", "Debil", 4, 1),
    ])
    _, current = elo.compute_elo(matches)
    r = current.set_index("team")["rating"]
    assert r["Fuerte"] > elo.BASE_RATING > r["Debil"]
    # El primero del ranking es el que gano todo.
    assert current.iloc[0]["team"] == "Fuerte"


def test_la_localia_mueve_los_numeros():
    # Mismo marcador 1-0; de local (no neutral) el local esperaba ganar mas,
    # asi que su premio por ganar es menor que si fuera en cancha neutral.
    base = _match("2020-01-01", "A", "B", 1, 0, neutral=False)
    _, local = elo.compute_elo(pd.DataFrame([base]))
    neutral = dict(base, neutral=True)
    _, neut = elo.compute_elo(pd.DataFrame([neutral]))
    gana_local_en_casa = local.set_index("team").loc["A", "rating"]
    gana_local_neutral = neut.set_index("team").loc["A", "rating"]
    assert gana_local_neutral > gana_local_en_casa


def test_los_partidos_importantes_mueven_mas_que_los_amistosos():
    upset = lambda cat: _match("2020-01-01", "A", "B", 1, 0, neutral=True, category=cat)
    _, mundial = elo.compute_elo(pd.DataFrame([upset("world_cup")]))
    _, amistoso = elo.compute_elo(pd.DataFrame([upset("friendly")]))
    cambio_mundial = mundial.set_index("team").loc["A", "rating"] - elo.BASE_RATING
    cambio_amistoso = amistoso.set_index("team").loc["A", "rating"] - elo.BASE_RATING
    assert cambio_mundial > cambio_amistoso
