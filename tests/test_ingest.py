"""Tests de la ingesta: clasificacion de torneos y normalizacion. Sin red."""

import numpy as np
import pandas as pd

from data import ingest


def test_clasificacion_de_torneos():
    assert ingest.classify_tournament("FIFA World Cup") == "world_cup"
    # Una clasificatoria NO es el torneo en si, aunque comparta nombre.
    assert ingest.classify_tournament("FIFA World Cup qualification") == "qualifier"
    assert ingest.classify_tournament("UEFA Euro") == "continental"
    assert ingest.classify_tournament("UEFA Euro qualification") == "qualifier"
    assert ingest.classify_tournament("UEFA Nations League") == "nations_league"
    assert ingest.classify_tournament("Copa América") == "continental"
    assert ingest.classify_tournament("Friendly") == "friendly"
    assert ingest.classify_tournament("Kirin Cup") == "other"
    assert ingest.classify_tournament(np.nan) == "other"


def test_normalizacion_basica():
    raw = pd.DataFrame([
        # jugado, no neutral, con seleccion historica a unificar
        {"date": "1990-06-10", "home_team": "West Germany", "away_team": "Yugoslavia",
         "home_score": 4, "away_score": 1, "tournament": "FIFA World Cup",
         "city": "Milan", "country": "Italy", "neutral": "TRUE"},
        # sin jugar (NA): debe descartarse
        {"date": "2026-06-27", "home_team": "Panama", "away_team": "England",
         "home_score": np.nan, "away_score": np.nan, "tournament": "FIFA World Cup",
         "city": "East Rutherford", "country": "United States", "neutral": "TRUE"},
        # amistoso normal
        {"date": "1985-01-01", "home_team": "Brazil", "away_team": "Chile",
         "home_score": 2, "away_score": 0, "tournament": "Friendly",
         "city": "Rio", "country": "Brazil", "neutral": "FALSE"},
    ])
    out = ingest.normalize_results(raw)

    # El partido sin jugar se cae.
    assert len(out) == 2
    # Orden cronologico.
    assert out["date"].is_monotonic_increasing
    # Selecciones historicas unificadas.
    assert "West Germany" not in set(out["home_team"])
    assert "Germany" in set(out["home_team"])
    assert "Serbia" in set(out["away_team"])
    # Tipos correctos.
    assert out["home_score"].dtype == int
    assert out["neutral"].dtype == bool
    assert out.loc[out["tournament"] == "Friendly", "neutral"].iloc[0] == False  # noqa: E712
    # Categoria anadida.
    assert set(out["tournament_category"]) == {"world_cup", "friendly"}
