"""Tests de validacion: metricas y harness de backtesting.

Las metricas se prueban con valores conocidos. El harness se prueba con datos
sinteticos (sin red, sin depender de los Parquet descargados).
"""

import numpy as np
import pandas as pd
import pytest

from validation import backtest, metrics


# --- metricas (valores conocidos) ------------------------------------------

def test_rps_casos_conocidos():
    # Prediccion perfecta -> 0.
    assert metrics.ranked_probability_score([1, 0, 0], 0) == pytest.approx(0.0)
    # Predecir visitante cuando gano el local -> el peor caso, 1.0.
    assert metrics.ranked_probability_score([0, 0, 1], 0) == pytest.approx(1.0)
    # Uniforme cuando gano el local -> 5/18.
    assert metrics.ranked_probability_score([1 / 3, 1 / 3, 1 / 3], 0) == pytest.approx(5 / 18)


def test_rps_respeta_el_orden():
    # Si gano el local, equivocarse por "empate" penaliza menos que por "visitante".
    casi = metrics.ranked_probability_score([0, 1, 0], 0)   # predijo empate
    lejos = metrics.ranked_probability_score([0, 0, 1], 0)  # predijo visitante
    assert casi < lejos


def test_log_loss_y_brier_conocidos():
    assert metrics.log_loss([0.5, 0.3, 0.2], 1) == pytest.approx(-np.log(0.3))
    assert metrics.brier_score([0.5, 0.3, 0.2], 1) == pytest.approx(0.5**2 + 0.7**2 + 0.2**2)


def test_un_modelo_informativo_le_gana_al_uniforme():
    # El favorito siempre gana y el modelo lo predice con confianza.
    preds = np.tile([0.8, 0.15, 0.05], (50, 1))
    outcomes = np.zeros(50, dtype=int)
    modelo = metrics.summarize(preds, outcomes)
    uniforme = metrics.summarize(np.tile([1 / 3, 1 / 3, 1 / 3], (50, 1)), outcomes)
    assert modelo["rps"] < uniforme["rps"]
    assert modelo["log_loss"] < uniforme["log_loss"]


def test_calibration_table_estructura():
    rng = np.random.default_rng(0)
    preds = rng.dirichlet([1, 1, 1], size=100)
    outcomes = rng.integers(0, 3, 100)
    table = metrics.calibration_table(preds, outcomes, n_bins=10)
    assert set(table.columns) >= {"mean_predicted", "observed_frequency", "count"}
    assert table["count"].sum() == 3 * 100  # cada partido aporta 3 pronosticos binarios
    assert (table["observed_frequency"].between(0, 1)).all()


# --- harness de backtesting (sintetico) ------------------------------------

def _synthetic_history(seed=0):
    """Historial sintetico: muchos amistosos antes de un 'Mundial 2099'."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    strength = {t: rng.normal(0, 0.4) for t in teams}
    rows = []
    dates = pd.date_range("2095-01-01", periods=400, freq="3D")
    di = 0
    for _ in range(14):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = np.exp(0.1 + 0.2 + strength[h] - strength[a])
                mu = np.exp(0.1 + strength[a] - strength[h])
                rows.append({
                    "date": dates[di % len(dates)], "home_team": h, "away_team": a,
                    "home_score": int(rng.poisson(lam)), "away_score": int(rng.poisson(mu)),
                    "neutral": False, "tournament": "Friendly", "tournament_category": "friendly",
                })
                di += 1
    # Torneo de prueba.
    for k, (h, a) in enumerate([("T0", "T1"), ("T2", "T3"), ("T0", "T2"), ("T4", "T5")]):
        rows.append({
            "date": pd.Timestamp(f"2099-06-{10 + k}"), "home_team": h, "away_team": a,
            "home_score": 2, "away_score": 0, "neutral": True,
            "tournament": "FIFA World Cup", "tournament_category": "world_cup",
        })
    return pd.DataFrame(rows)


def test_backtest_tournament_mecanica():
    matches = _synthetic_history()
    fit_kwargs = {"min_matches": 0, "l2": 0.5, "half_life_days": None, "max_age_years": None}
    result = backtest.backtest_tournament(matches, "FIFA World Cup", 2099, fit_kwargs=fit_kwargs)

    assert result["outcomes"].tolist() == [0, 0, 0, 0]  # todos los locales ganaron 2-0
    for predictor in ("model", "uniform", "elo"):
        preds = result["predictions"][predictor]
        assert preds.shape == (4, 3)
        assert np.allclose(preds.sum(axis=1), 1.0)       # cada prediccion suma 1
    assert np.allclose(result["predictions"]["uniform"], 1 / 3)
    # El entrenamiento es estricto: nada del torneo 2099 entra al train.
    assert result["cutoff"].startswith("2099")


def test_run_backtest_agrega_y_calibra():
    matches = _synthetic_history()
    fit_kwargs = {"min_matches": 0, "l2": 0.5, "half_life_days": None, "max_age_years": None}
    summary, calibration, pooled = backtest.run_backtest(
        matches, tournaments=[{"name": "FIFA World Cup", "year": 2099, "label": "Mundial 2099"}],
        fit_kwargs=fit_kwargs,
    )
    assert set(summary["predictor"]) == {"model", "uniform", "elo"}
    assert "Todos" in set(summary["tournament"])
    assert not calibration.empty
    assert len(pooled["outcomes"]) == 4
