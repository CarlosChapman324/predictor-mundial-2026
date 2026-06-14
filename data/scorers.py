"""Ingesta de goleadores internacionales y reparto de goles por jugador (Capa 2).

Fuente: goalscorers.csv del mismo repo del historico (martj42), con el autor de
cada gol internacional desde 1916. Sirve para estimar, dentro de cada seleccion,
que fraccion de los goles suele meter cada jugador (su "cuota"). Es la base del
submodelo de goleadores y de la Bota de Oro.

CAPA 2 (experimental): a diferencia del modelo de goles, estos datos no traen
los convocados del 2026 ni los minutos jugados, asi que se usa a los goleadores
recientes como proxy del plantel. Confianza baja, por eso se etiqueta siempre.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"


def download_goalscorers(raw_dir: Path | str, *, timeout: int = 60) -> Path:
    """Descarga goalscorers.csv a raw_dir y devuelve la ruta."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "goalscorers.csv"
    response = requests.get(GOALSCORERS_URL, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def player_shares(goalscorers: pd.DataFrame, *, since=None) -> pd.DataFrame:
    """Cuota de goles de cada jugador dentro de su seleccion.

    share = goles del jugador / goles totales de su seleccion (en la ventana).
    Excluye autogoles. La suma de cuotas por seleccion es 1, asi que share es
    P(lo marca este jugador | la seleccion marca un gol).

    since : fecha (str o Timestamp) para limitar a un periodo reciente.
    """
    g = goalscorers.copy()
    g = g[~g["own_goal"].fillna(False).astype(bool)]
    if since is not None:
        g = g[pd.to_datetime(g["date"]) >= pd.to_datetime(since)]

    counts = g.groupby(["team", "scorer"]).size().rename("goals").reset_index()
    team_total = counts.groupby("team")["goals"].transform("sum")
    counts["share"] = counts["goals"] / team_total
    return counts.sort_values(["team", "goals"], ascending=[True, False]).reset_index(drop=True)
