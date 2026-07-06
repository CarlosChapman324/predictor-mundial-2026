"""Ingesta y normalizacion del historico de partidos internacionales.

Fuente: dataset publico de resultados internacionales desde 1872 (repo martj42),
el mismo que circula en Kaggle pero sin necesidad de autenticarse. Trae fecha,
equipos, marcador, torneo, ciudad, pais y si se jugo en cancha neutral.

Separamos claramente:
  - descarga (toca la red)            -> download_results
  - normalizacion (matematica/datos)  -> normalize_results  (pura, sin red)

El modelo nunca importa de aqui en tiempo de ejecucion: consume el Parquet ya
guardado. Por eso la normalizacion es una funcion pura y testeable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# URL cruda del CSV principal. Lo dejamos como constante para que sea facil de
# auditar y de cambiar si la fuente se mueve.
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

# ---------------------------------------------------------------------------
# Continuidad de selecciones (decision de modelado, configurable)
# ---------------------------------------------------------------------------
# Mapeamos estados desaparecidos a su sucesor reconocido por la FIFA para que el
# Elo herede la historia completa. Es discutible y por eso queda explicito y
# documentado: cualquiera puede desactivarlo. Solo incluimos las continuidades
# claras; lo dudoso se deja como esta.
TEAM_NAME_MAP = {
    "West Germany": "Germany",          # RFA es la continuadora; Alemania reunificada hereda su historia
    "Czechoslovakia": "Czech Republic",  # sucesor futbolistico que juega el 2026
    "Soviet Union": "Russia",            # la FIFA asigna el palmares de la URSS a Rusia
    "Yugoslavia": "Serbia",              # idem: Serbia es la continuadora de la RFS de Yugoslavia
    "Serbia and Montenegro": "Serbia",
    "Zaire": "DR Congo",                 # mismo pais, cambio de nombre
}

# ---------------------------------------------------------------------------
# Categorias de torneo por importancia
# ---------------------------------------------------------------------------
# Sirven para dos cosas: el factor K del Elo (un Mundial pesa mas que un
# amistoso) y, mas adelante, la ponderacion por relevancia en la verosimilitud
# del modelo de goles. Guardamos la CATEGORIA en el Parquet; el mapeo a numeros
# vive en la capa que los usa (model/elo.py), para no mezclar datos con modelo.
CONTINENTAL_TOURNAMENTS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "OFC Nations Cup",
}


def classify_tournament(name: str) -> str:
    """Asigna una categoria de importancia a partir del nombre del torneo.

    El orden de las reglas importa: 'UEFA Euro qualification' contiene
    'UEFA Euro', asi que primero detectamos las clasificatorias.
    """
    if not isinstance(name, str):
        return "other"
    if "qualification" in name:
        return "qualifier"
    if name in CONTINENTAL_TOURNAMENTS:
        return "continental"
    if "Nations League" in name:
        return "nations_league"
    if name == "FIFA World Cup":
        return "world_cup"
    if "Confederations" in name:
        return "confederations"
    if name == "Friendly":
        return "friendly"
    return "other"


def download_results(raw_dir: Path | str, *, timeout: int = 60) -> Path:
    """Descarga results.csv a raw_dir y devuelve la ruta. Unica funcion con red."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "results.csv"
    response = requests.get(RESULTS_URL, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_shootouts(raw_dir: Path | str, *, timeout: int = 60) -> Path:
    """Descarga shootouts.csv (ganadores por penales) a raw_dir. Con red."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "shootouts.csv"
    response = requests.get(SHOOTOUTS_URL, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_raw(csv_path: Path | str) -> pd.DataFrame:
    """Lee el CSV crudo sin transformar (incluye partidos sin jugar, score NA)."""
    return pd.read_csv(csv_path)


def normalize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpia el historico y lo deja listo para el modelo y el Elo.

    Pasos:
      1. fecha a datetime
      2. descartar partidos sin jugar (marcador NA): no aportan al modelo
      3. marcador a entero
      4. 'neutral' a booleano de verdad
      5. unificar nombres de selecciones desaparecidas (TEAM_NAME_MAP)
      6. anadir la categoria de torneo
      7. ordenar cronologicamente (imprescindible para el Elo, que es secuencial)
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    played = df["home_score"].notna() & df["away_score"].notna()
    df = df.loc[played].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["neutral"] = df["neutral"].map(_to_bool).astype(bool)

    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    df["tournament_category"] = df["tournament"].map(classify_tournament)

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    columns = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "tournament_category", "city", "country", "neutral",
    ]
    return df[columns]


def _to_bool(value) -> bool:
    """'TRUE'/'FALSE' (u otras variantes) a booleano de Python."""
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "TRUE"


def save_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Guarda un DataFrame en Parquet, creando la carpeta si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
