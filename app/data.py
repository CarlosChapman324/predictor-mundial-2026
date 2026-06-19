"""Carga de datos para el dashboard. Solo LEE de disco; no calcula el modelo.

Cada archivo se cachea con st.cache_data. Si falta alguno, las funciones
devuelven None y la app muestra un aviso de que hay que correr el script
correspondiente, en vez de romperse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SNAPSHOT = ROOT / "data" / "snapshot"  # congelado versionado para el deploy
REFERENCE = ROOT / "data" / "reference"


def _resolve(name: str) -> Path | None:
    """Busca un archivo primero en processed (local) y luego en snapshot (deploy)."""
    for base in (PROCESSED, SNAPSHOT):
        path = base / name
        if path.exists():
            return path
    return None


def _read_parquet(name: str):
    path = _resolve(name)
    return pd.read_parquet(path) if path is not None else None


def _read_processed_json(name: str):
    path = _resolve(name)
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False, ttl=3600)
def simulation():
    return _read_parquet("simulation_probabilities.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def match_markets():
    df = _read_parquet("match_markets.parquet")
    if df is not None:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def elo():
    return _read_parquet("elo_current.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def strengths():
    return _read_parquet("goal_model_strengths.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def backtest_summary():
    return _read_parquet("backtest_summary.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def backtest_calibration():
    return _read_parquet("backtest_calibration.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def market_comparison():
    return _read_parquet("market_comparison.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def groups():
    data = _read_json(REFERENCE / "groups.json")
    return data["grupos"] if data else None


@st.cache_data(show_spinner=False, ttl=3600)
def model_params():
    return _read_processed_json("goal_model_params.json")


@st.cache_data(show_spinner=False, ttl=3600)
def simulation_meta():
    return _read_processed_json("simulation_meta.json")


@st.cache_data(show_spinner=False, ttl=3600)
def predictions_history():
    return _read_parquet("predictions_history.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def golden_boot():
    return _read_parquet("golden_boot.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def match_scorers():
    return _read_parquet("match_scorers.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def props_predictions():
    return _read_parquet("props_predictions.parquet")


@st.cache_data(show_spinner=False, ttl=3600)
def value_analysis():
    return _read_parquet("value_analysis.parquet")
