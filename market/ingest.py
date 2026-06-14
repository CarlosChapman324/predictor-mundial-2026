"""Ingesta de cuotas de casas de apuestas.

Dos vias, porque las cuotas reales requieren una fuente externa:
  - The Odds API (https://the-odds-api.com): se activa con una API key (plan
    gratis, 500 requests/mes). Trae cuotas en vivo de varios bookmakers.
  - CSV local: para cuotas que el usuario consiga por su cuenta, con un esquema
    simple y documentado.

Es la unica parte del modulo de mercado que toca la red. La consolidacion de
varias casas en un "consenso" se hace promediando sus probabilidades implicitas
(ya sin margen), que es mas robusto que promediar cuotas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from market import odds as odds_module

THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
WORLD_CUP_SPORT_KEY = "soccer_fifa_world_cup"

# Esquema esperado de un CSV de cuotas local.
CSV_COLUMNS = ["home_team", "away_team", "home_odds", "draw_odds", "away_odds"]


def fetch_the_odds_api(
    api_key: str,
    *,
    sport: str = WORLD_CUP_SPORT_KEY,
    regions: str = "eu",
    markets: str = "h2h",
    odds_format: str = "decimal",
    timeout: int = 30,
) -> pd.DataFrame:
    """Descarga cuotas 1X2 de The Odds API y las normaliza a formato largo.

    Devuelve una fila por (partido, casa) con home_odds, draw_odds, away_odds.
    """
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    response = requests.get(THE_ODDS_API_URL.format(sport=sport), params=params, timeout=timeout)
    response.raise_for_status()
    return _normalize_events(response.json())


def _normalize_events(events: list[dict]) -> pd.DataFrame:
    """Aplana la respuesta (eventos -> casas -> mercado h2h -> resultados)."""
    rows = []
    for event in events:
        home, away = event.get("home_team"), event.get("away_team")
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "commence_time": event.get("commence_time"),
                    "bookmaker": book.get("key"),
                    "home_odds": prices.get(home),
                    "draw_odds": prices.get("Draw"),
                    "away_odds": prices.get(away),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["home_odds", "draw_odds", "away_odds"])
    return df


def load_odds_csv(path: Path | str) -> pd.DataFrame:
    """Carga cuotas desde un CSV local con el esquema CSV_COLUMNS."""
    df = pd.read_csv(path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Al CSV de cuotas le faltan columnas: {missing}")
    return df


def consensus_market(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Consolida varias casas en una probabilidad de mercado de consenso.

    Para cada partido promedia las probabilidades implicitas (sin margen) de
    todas las casas disponibles. Devuelve una fila por partido con
    market_home/draw/away y el overround medio observado.
    """
    if odds_df.empty:
        return pd.DataFrame(columns=["home_team", "away_team", "market_home", "market_draw", "market_away", "overround", "n_books"])

    with_probs = odds_module.add_market_probabilities(odds_df)
    grouped = with_probs.groupby(["home_team", "away_team"], as_index=False).agg(
        market_home=("market_home", "mean"),
        market_draw=("market_draw", "mean"),
        market_away=("market_away", "mean"),
        overround=("overround", "mean"),
        n_books=("market_home", "size"),
    )
    # Renormaliza por si el promedio de probabilidades se desvia ligeramente de 1.
    total = grouped[["market_home", "market_draw", "market_away"]].sum(axis=1)
    for col in ["market_home", "market_draw", "market_away"]:
        grouped[col] = grouped[col] / total
    return grouped
