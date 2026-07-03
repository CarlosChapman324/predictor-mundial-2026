"""Tests del fixture 2026. Usa los datos curados reales (sin red) y datos
sinteticos para la construccion del calendario."""

from pathlib import Path

import pandas as pd
import pytest

from data import fixture

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


def test_grupos_reales_cumplen_el_formato():
    groups = fixture.load_groups(REFERENCE_DIR)
    assert groups["group"].nunique() == 12
    assert (groups.groupby("group")["team"].size() == 4).all()
    assert groups["team"].nunique() == 48
    assert not groups["team"].duplicated().any()


def test_sedes_reales():
    venues = fixture.load_venues(REFERENCE_DIR)
    assert len(venues) == 16
    assert venues["dataset_city"].is_unique
    assert set(venues["country"]) == {"United States", "Canada", "Mexico"}


def test_construccion_del_calendario_y_join_de_sede():
    # Un grupo sintetico de 4 equipos -> sus 6 partidos, en una sede real.
    teams = ["T1", "T2", "T3", "T4"]
    groups = pd.DataFrame(
        [{"group": "A", "seed_position": i + 1, "team": t} for i, t in enumerate(teams)]
    )
    venues = fixture.load_venues(REFERENCE_DIR)

    pairings = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]  # 6 pares
    rows = []
    for k, (h, a) in enumerate(pairings):
        played = k < 2  # los dos primeros, jugados
        rows.append({
            "tournament": "FIFA World Cup", "date": f"2026-06-1{k}",
            "home_team": h, "away_team": a,
            "home_score": 1 if played else None, "away_score": 0 if played else None,
            "city": "Mexico City",
        })
    raw = pd.DataFrame(rows)

    schedule = fixture.build_group_stage_schedule(raw, groups, venues)
    assert len(schedule) == 6
    assert (schedule["group"] == "A").all()
    assert schedule["stadium"].eq("Estadio Azteca").all()  # join por ciudad
    assert int(schedule["played"].sum()) == 2


def test_validacion_falla_si_falta_un_grupo():
    groups = fixture.load_groups(REFERENCE_DIR)
    venues = fixture.load_venues(REFERENCE_DIR)
    roto = groups[groups["group"] != "L"]  # solo 11 grupos
    with pytest.raises(AssertionError):
        fixture.validate_fixture(roto, venues, pd.DataFrame())


def test_excluye_partidos_de_eliminatoria():
    # Cuando arranca la eliminatoria, el historico trae cruces entre grupos
    # distintos; el calendario de grupos debe quedarse solo con los del mismo grupo.
    groups = pd.DataFrame(
        [{"group": "A", "seed_position": i + 1, "team": t} for i, t in enumerate(["A1", "A2", "A3", "A4"])]
        + [{"group": "B", "seed_position": i + 1, "team": t} for i, t in enumerate(["B1", "B2", "B3", "B4"])]
    )
    venues = fixture.load_venues(REFERENCE_DIR)
    raw = pd.DataFrame([
        {"tournament": "FIFA World Cup", "date": "2026-06-12", "home_team": "A1", "away_team": "A2",
         "home_score": 1, "away_score": 0, "city": "Mexico City"},   # grupos (A vs A)
        {"tournament": "FIFA World Cup", "date": "2026-06-29", "home_team": "A1", "away_team": "B2",
         "home_score": 2, "away_score": 1, "city": "Mexico City"},   # eliminatoria (A vs B): se excluye
    ])
    schedule = fixture.build_group_stage_schedule(raw, groups, venues)
    assert len(schedule) == 1
    assert set(schedule["group"]) == {"A"}
