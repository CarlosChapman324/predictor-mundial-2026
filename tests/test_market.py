"""Tests del modulo de mercado: conversion de cuotas y comparacion. Sin red."""

import pandas as pd
import pytest

from market import ingest, odds


# --- conversion de cuotas a probabilidad implicita -------------------------

def test_implied_probabilities_quita_el_margen():
    # Cuotas justas (las inversas ya suman 1): la conversion no las altera.
    p = odds.implied_probabilities([2.0, 4.0, 4.0])
    assert list(p) == pytest.approx([0.5, 0.25, 0.25])
    assert float(p.sum()) == pytest.approx(1.0)


def test_overround_mide_el_margen():
    assert odds.overround([2.0, 4.0, 4.0]) == pytest.approx(0.0)        # sin margen
    assert odds.overround([1.9, 1.9]) == pytest.approx(2 / 1.9 - 1)     # ~5.3%


def test_la_implicita_favorece_la_cuota_baja():
    p = odds.implied_probabilities([1.5, 4.0, 7.0])
    assert p[0] > p[1] > p[2]


def test_add_market_probabilities():
    df = pd.DataFrame([{"home_odds": 2.0, "draw_odds": 4.0, "away_odds": 4.0}])
    out = odds.add_market_probabilities(df)
    assert out["market_home"].iloc[0] == pytest.approx(0.5)
    assert out["overround"].iloc[0] == pytest.approx(0.0)


def test_add_edges_y_favoritos():
    df = pd.DataFrame([{
        "model_home": 0.6, "model_draw": 0.25, "model_away": 0.15,
        "market_home": 0.5, "market_draw": 0.30, "market_away": 0.20,
    }])
    out = odds.add_edges(df)
    assert out["edge_home"].iloc[0] == pytest.approx(0.1)
    assert out["model_favorite"].iloc[0] == "home"
    assert out["market_favorite"].iloc[0] == "home"
    assert bool(out["same_favorite"].iloc[0]) is True
    assert out["max_abs_edge"].iloc[0] == pytest.approx(0.1)


def test_efficiency_summary_compara_quien_predijo_mejor():
    df = pd.DataFrame([
        {"model_home": 0.6, "model_draw": 0.25, "model_away": 0.15,
         "market_home": 0.5, "market_draw": 0.30, "market_away": 0.20, "overround": 0.05, "outcome": 0},
        {"model_home": 0.2, "model_draw": 0.30, "model_away": 0.5,
         "market_home": 0.25, "market_draw": 0.30, "market_away": 0.45, "overround": 0.05, "outcome": 2},
    ])
    df = odds.add_edges(df)
    summary = odds.efficiency_summary(df, outcome_col="outcome")
    assert summary["n"] == 2
    assert summary["avg_overround"] == pytest.approx(0.05)
    assert 0.0 <= summary["agreement_favorite"] <= 1.0
    assert "rps_model" in summary and "rps_market" in summary


# --- ingesta ---------------------------------------------------------------

def test_consensus_promedia_varias_casas():
    odds_df = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "bookmaker": "x", "home_odds": 2.0, "draw_odds": 3.5, "away_odds": 3.8},
        {"home_team": "A", "away_team": "B", "bookmaker": "y", "home_odds": 2.1, "draw_odds": 3.4, "away_odds": 3.6},
    ])
    consensus = ingest.consensus_market(odds_df)
    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["n_books"] == 2
    assert row[["market_home", "market_draw", "market_away"]].sum() == pytest.approx(1.0)


def test_load_odds_csv_valida_el_esquema(tmp_path):
    good = tmp_path / "odds.csv"
    pd.DataFrame([{"home_team": "A", "away_team": "B",
                   "home_odds": 2.0, "draw_odds": 3.0, "away_odds": 4.0}]).to_csv(good, index=False)
    assert len(ingest.load_odds_csv(good)) == 1

    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"home_team": "A"}]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        ingest.load_odds_csv(bad)
