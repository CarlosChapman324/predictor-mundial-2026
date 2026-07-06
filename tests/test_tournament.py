"""Tests del motor del torneo: formato 2026, desempates y Monte Carlo. Sin red."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import fixture as fixture_loader
from model.goals import FittedGoalModel
from tournament import format2026, montecarlo

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


# --- desempates 2026 (la parte critica) ------------------------------------

def test_desempate_head_to_head_va_antes_que_diferencia_global():
    # A y B empatan a 6 puntos. B tiene MEJOR diferencia de goles global, pero A
    # le gano en el head-to-head. Con la regla 2026, A debe quedar por encima
    # (con la regla vieja, basada en diferencia global, quedaria B).
    results = [
        ("A", 1, "B", 0),  # A gana el enfrentamiento directo
        ("A", 1, "C", 0),
        ("D", 1, "A", 0),
        ("B", 4, "C", 0),  # B golea para inflar su diferencia global
        ("B", 3, "D", 0),
        ("C", 2, "D", 0),
    ]
    teams = ["A", "B", "C", "D"]
    ranked, stats = format2026.rank_group(teams, results)

    # El test es significativo: B de verdad tiene mejor diferencia global que A.
    assert stats["B"]["gd"] > stats["A"]["gd"]
    assert stats["A"]["points"] == stats["B"]["points"] == 6
    # Y aun asi, por head-to-head, A va primero.
    assert ranked[:2] == ["A", "B"]
    assert ranked == ["A", "B", "C", "D"]


def test_orden_simple_por_puntos():
    results = [
        ("A", 3, "B", 0), ("A", 2, "C", 0), ("A", 1, "D", 0),
        ("B", 2, "C", 0), ("B", 2, "D", 0), ("C", 1, "D", 0),
    ]
    ranked, _ = format2026.rank_group(["A", "B", "C", "D"], results)
    assert ranked == ["A", "B", "C", "D"]


# --- mejores terceros y asignacion de llaves -------------------------------

def test_seleccion_de_mejores_terceros():
    thirds = [
        {"group": g, "team": f"3{g}", "stats": {"points": p, "gd": d, "gf": f}}
        for g, p, d, f in [
            ("A", 6, 3, 5), ("B", 4, 1, 3), ("C", 4, 2, 4), ("D", 3, 0, 2),
            ("E", 7, 4, 6), ("F", 2, -1, 1), ("G", 5, 2, 5), ("H", 4, 0, 2),
            ("I", 3, 1, 3), ("J", 6, 2, 4), ("K", 1, -3, 1), ("L", 4, 1, 2),
        ]
    ]
    best = format2026.select_best_thirds(thirds, n=8)
    assert len(best) == 8
    chosen = {t["group"] for t in best}
    # Los dos peores por puntos (F=2, K=1) deben quedar fuera.
    assert "F" not in chosen and "K" not in chosen
    # El mejor por puntos (E=7) debe entrar.
    assert "E" in chosen


def test_asignacion_de_terceros_evita_el_propio_grupo():
    bracket = format2026.load_bracket(REFERENCE_DIR)
    qualified = [{"group": g, "team": f"3{g}"} for g in ["A", "B", "E", "I", "D", "G", "L", "K"]]
    assignment = format2026.assign_thirds_to_slots(qualified, bracket["third_slots"])
    assert len(assignment) == 8
    # Ningun tercero enfrenta al ganador de su propio grupo.
    faced = {slot["match"]: slot["faced_group"] for slot in bracket["third_slots"]}
    team_group = {f"3{g}": g for g in ["A", "B", "E", "I", "D", "G", "L", "K"]}
    for match, team in assignment.items():
        assert team_group[team] != faced[match]


def test_estructura_del_cuadro():
    bracket = format2026.load_bracket(REFERENCE_DIR)
    assert len(bracket["round_of_32"]) == 16
    assert len(bracket["third_slots"]) == 8
    # 8 octavos + 4 cuartos + 2 semis + 1 final = 15 cruces en el arbol.
    assert len(bracket["tree"]) == 15
    assert bracket["rounds"]["final"] == [104]


# --- Monte Carlo (integracion, autocontenido) ------------------------------

def _synthetic_fixture(groups: pd.DataFrame) -> pd.DataFrame:
    """Calendario sintetico de fase de grupos: todos contra todos, sin jugar."""
    rows = []
    for group, block in groups.groupby("group"):
        teams = list(block["team"])
        for home, away in combinations(teams, 2):
            rows.append({
                "group": group, "home_team": home, "away_team": away,
                "home_score": np.nan, "away_score": np.nan,
                "played": False, "country": "Neutral",
            })
    return pd.DataFrame(rows)


def _hand_model(teams) -> FittedGoalModel:
    # Fuerzas en gradiente para que las probabilidades no sean degeneradas.
    teams = sorted(teams)
    attack = {t: 0.4 - 0.8 * i / (len(teams) - 1) for i, t in enumerate(teams)}
    defense = {t: -0.4 + 0.8 * i / (len(teams) - 1) for i, t in enumerate(teams)}
    return FittedGoalModel(intercept=0.1, home_advantage=0.2, rho=-0.05,
                           attack=attack, defense=defense)


def test_monte_carlo_cumple_los_invariantes_del_torneo():
    groups = fixture_loader.load_groups(REFERENCE_DIR)
    bracket = format2026.load_bracket(REFERENCE_DIR)
    fx = _synthetic_fixture(groups)
    model = _hand_model(groups["team"])

    n = 400
    probs = montecarlo.run_monte_carlo(fx, model, bracket, n_sims=n, seed=0)

    assert len(probs) == 48
    # Cada simulacion produce exactamente 1 campeon, 2 finalistas, 4 semifinalistas,
    # 8 en cuartos, 16 en octavos, 32 clasificados, 12 primeros, 12 segundos, 8 terceros.
    assert probs["champion"].sum() == pytest.approx(1.0, abs=1e-9)
    assert probs["final"].sum() == pytest.approx(2.0, abs=1e-9)
    assert probs["semifinal"].sum() == pytest.approx(4.0, abs=1e-9)
    assert probs["quarterfinal"].sum() == pytest.approx(8.0, abs=1e-9)
    assert probs["round_of_16"].sum() == pytest.approx(16.0, abs=1e-9)
    assert probs["qualify"].sum() == pytest.approx(32.0, abs=1e-9)
    assert probs["win_group"].sum() == pytest.approx(12.0, abs=1e-9)
    assert probs["runner_up"].sum() == pytest.approx(12.0, abs=1e-9)
    assert probs["third"].sum() == pytest.approx(8.0, abs=1e-9)

    # Monotonia: campeon <= final <= ... <= clasificar, por equipo.
    for _, r in probs.iterrows():
        assert r["champion"] <= r["final"] <= r["semifinal"] <= r["quarterfinal"]
        assert r["quarterfinal"] <= r["round_of_16"] <= r["qualify"] <= 1.0


def test_simular_un_torneo_da_un_campeon_y_32_clasificados():
    groups = fixture_loader.load_groups(REFERENCE_DIR)
    bracket = format2026.load_bracket(REFERENCE_DIR)
    fx = _synthetic_fixture(groups)
    model = _hand_model(groups["team"])

    rng = np.random.default_rng(7)
    positions, qualified, stage, champion = montecarlo.simulate_tournament(
        fx, model, bracket, rng=rng
    )
    assert len(positions) == 48          # los 48 reciben posicion de grupo
    assert len(qualified) == 32          # 24 primeros/segundos + 8 terceros
    assert champion in qualified
    assert stage[champion] == montecarlo.STAGE_CHAMPION


def test_ignore_played_trata_los_jugados_como_pendientes():
    # Capa viva: con ignore_played=True, un partido ya jugado se vuelve a simular
    # (no se fija su resultado), util para el pronostico sin condicionar.
    groups = fixture_loader.load_groups(REFERENCE_DIR)
    fx = _synthetic_fixture(groups)
    fx.loc[0, ["played", "home_score", "away_score"]] = [True, 2, 0]
    model = _hand_model(groups["team"])

    normal = montecarlo._prepare_group_matches(fx, model, 10, ignore_played=False)
    ignored = montecarlo._prepare_group_matches(fx, model, 10, ignore_played=True)
    assert "fixed" in normal[0]       # condicionado: resultado fijo
    assert "fixed" not in ignored[0]  # sin condicionar: se muestrea


def test_condiciona_la_eliminatoria_jugada():
    # Capa viva del cuadro: un equipo que perdio un cruce jugado queda eliminado
    # y no puede ser campeon, aunque el cuadro reconstruido lo empareje distinto.
    groups = fixture_loader.load_groups(REFERENCE_DIR)
    bracket = format2026.load_bracket(REFERENCE_DIR)
    fx = _synthetic_fixture(groups)
    model = _hand_model(groups["team"])
    teams = sorted(groups["team"])
    winner, loser = teams[0], teams[1]
    played_knockout = {frozenset((winner, loser)): winner}

    probs = montecarlo.run_monte_carlo(
        fx, model, bracket, n_sims=200, seed=0, played_knockout=played_knockout
    ).set_index("team")
    assert probs.loc[loser, "champion"] == 0.0   # eliminado: nunca campeon
    assert probs["champion"].sum() == pytest.approx(1.0)
