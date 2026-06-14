"""Tests del modelo de goles (Poisson + Dixon-Coles). Matematica pura, sin red."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import gammaln

from model import goals
from model.markets import one_x_two


# --- matriz de marcadores --------------------------------------------------

def test_la_matriz_es_una_distribucion_valida():
    m = goals.score_matrix(1.6, 1.1, rho=-0.1, max_goals=10)
    assert m.shape == (11, 11)
    assert (m >= 0).all()
    assert m.sum() == pytest.approx(1.0)


def test_sin_rho_coincide_con_poisson_independiente():
    lh, la, mg = 1.7, 1.2, 10
    m = goals.score_matrix(lh, la, rho=0.0, max_goals=mg)
    k = np.arange(mg + 1)
    pmf_h = np.exp(k * np.log(lh) - lh - gammaln(k + 1))
    pmf_a = np.exp(k * np.log(la) - la - gammaln(k + 1))
    independent = np.outer(pmf_h, pmf_a)
    independent /= independent.sum()
    assert np.allclose(m, independent)


def test_dixon_coles_sube_los_empates_bajos():
    # rho negativo debe subir 0-0 y 1-1 (y por tanto la probabilidad de empate).
    base = goals.score_matrix(1.3, 1.3, rho=0.0)
    dc = goals.score_matrix(1.3, 1.3, rho=-0.15)
    assert dc[0, 0] > base[0, 0]
    assert dc[1, 1] > base[1, 1]
    assert one_x_two(dc)["draw"] > one_x_two(base)["draw"]


def test_mas_goles_esperados_del_local_favorecen_al_local():
    fuerte_local = goals.score_matrix(2.2, 0.8, rho=-0.1)
    r = one_x_two(fuerte_local)
    assert r["home"] > r["away"]


# --- modelo ajustado (objeto) ----------------------------------------------

def test_expected_goals_y_localia():
    model = goals.FittedGoalModel(
        intercept=0.0, home_advantage=0.3, rho=-0.1,
        attack={"A": 0.5, "B": -0.2}, defense={"A": -0.1, "B": 0.2},
    )
    lh, la = model.expected_goals("A", "B", home_advantage=True)
    assert lh == pytest.approx(np.exp(0.3 + 0.5 + 0.2))
    assert la == pytest.approx(np.exp(-0.2 - 0.1))
    assert lh > la
    # Quitar la localia baja el lambda del local.
    lh_neutral, _ = model.expected_goals("A", "B", home_advantage=False)
    assert lh_neutral < lh
    # Una seleccion desconocida se trata como promedio (ataque/defensa 0):
    # aqui la defensa del rival desconocido es 0, asi que el lambda del local
    # depende solo de su propio ataque (sin localia).
    lh2, _ = model.expected_goals("A", "Desconocido", home_advantage=False)
    assert lh2 == pytest.approx(np.exp(0.0 + 0.5 + 0.0))


# --- gradiente analitico ---------------------------------------------------

def _synthetic_data(seed=0):
    rng = np.random.default_rng(seed)
    n_teams = 4
    n = 60
    hi = rng.integers(0, n_teams, n)
    ai = (hi + rng.integers(1, n_teams, n)) % n_teams  # distinto del local
    return {
        "home_idx": hi, "away_idx": ai,
        "home_goals": rng.integers(0, 4, n).astype(float),
        "away_goals": rng.integers(0, 4, n).astype(float),
        "not_neutral": rng.integers(0, 2, n).astype(float),
        "weights": rng.uniform(0.5, 1.5, n),
    }, n_teams


def test_el_gradiente_analitico_coincide_con_el_numerico():
    data, n_teams = _synthetic_data()
    l2 = 0.7
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 0.1, 3 + 2 * n_teams)
    theta[2] = -0.05  # rho dentro de la cota

    f = lambda th: goals._negative_log_likelihood(th, data, n_teams, l2)[0]
    g = lambda th: goals._negative_log_likelihood(th, data, n_teams, l2)[1]
    error = check_grad(f, g, theta)
    assert error < 1e-4


# --- recuperacion de parametros (validacion del estimador) -----------------

def test_el_ajuste_recupera_las_fuerzas_conocidas():
    # Generamos partidos con fuerzas conocidas y comprobamos que el MLE las recupera.
    rng = np.random.default_rng(42)
    n_teams = 8
    teams = [f"T{i}" for i in range(n_teams)]
    true_attack = rng.normal(0, 0.35, n_teams)
    true_defense = rng.normal(0, 0.35, n_teams)
    intercept, home_adv = 0.1, 0.30

    rows = []
    for _ in range(12):  # varias rondas de todos contra todos, de local y visitante
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lam = np.exp(intercept + home_adv + true_attack[i] + true_defense[j])
                mu = np.exp(intercept + true_attack[j] + true_defense[i])
                rows.append({
                    "date": pd.Timestamp("2024-01-01"),
                    "home_team": teams[i], "away_team": teams[j],
                    "home_score": rng.poisson(lam), "away_score": rng.poisson(mu),
                    "neutral": False, "tournament_category": "world_cup",
                })
    matches = pd.DataFrame(rows)

    model = goals.fit_goal_model(
        matches, half_life_days=None, max_age_years=None,
        min_matches=0, l2=0.0,
    )

    # El ataque/defensa se identifica salvo una constante aditiva, asi que
    # comparamos por correlacion (invariante a esa constante).
    est_attack = np.array([model.attack[t] for t in teams])
    est_defense = np.array([model.defense[t] for t in teams])
    assert np.corrcoef(est_attack, true_attack)[0, 1] > 0.9
    assert np.corrcoef(est_defense, true_defense)[0, 1] > 0.9
    # La localia si esta identificada: debe quedar cerca del valor real.
    assert model.home_advantage == pytest.approx(home_adv, abs=0.1)
