"""Cliente de API-Football (acceso directo de API-Sports) con reintentos y cache.

Capa de red de la Capa 2. El modelo y los submodelos NO importan de aqui: solo
consumen los Parquet que deja la ingesta. Cachear es obligatorio porque el plan
free de API-Football da 100 requests/dia: cada respuesta se guarda en disco y las
llamadas repetidas con los mismos parametros NO vuelven a tocar la red.

La clave se lee de la variable de entorno APISPORTS_KEY (cargable desde .env con
load_env); nunca se escribe en el codigo ni se commitea (ver .env.example y los
secrets de GitHub Actions).

El parsing (parse_*) y la agregacion (referee_card_averages) son funciones puras
y testeables sin red, a partir de payloads ya descargados.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://v3.football.api-sports.io"  # acceso directo de API-Sports
API_KEY_ENV = "APISPORTS_KEY"
WORLD_CUP_LEAGUE_ID = 1  # FIFA World Cup en API-Football
CACHE_DIR = ROOT / "data" / "raw" / "apifootball_cache"
RETRY_STATUS = {429, 500, 502, 503, 504}


def load_env(path: Path | str | None = None) -> None:
    """Carga el .env (si existe) en el entorno, sin pisar lo ya definido.

    Asi los scripts encuentran APISPORTS_KEY en local; en el cron va por secrets.
    """
    env_path = Path(path) if path else ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Nombres de los tipos de estadistica en /fixtures/statistics -> nombre interno.
STAT_TYPES = {
    "Corner Kicks": "corners",
    "Total Shots": "shots",
    "Shots on Goal": "shots_on_target",
    "Fouls": "fouls",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
}


def _cache_path(path: str, params: dict) -> Path:
    key = json.dumps({"path": path, "params": params}, sort_keys=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def get(path: str, params: dict | None = None, *, api_key: str | None = None,
        use_cache: bool = True, max_retries: int = 4, backoff_base: float = 1.0,
        session=None) -> dict:
    """GET cacheado contra API-Football, con reintentos y backoff exponencial.

    Si la respuesta ya esta cacheada en disco, la devuelve sin tocar la red (asi
    se respeta el limite free). session permite inyectar un cliente simulado en
    los tests.
    """
    params = params or {}
    cache_file = _cache_path(path, params)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Falta {API_KEY_ENV} (define la variable de entorno o usa la cache).")

    headers = {"x-apisports-key": api_key}
    getter = session.get if session is not None else requests.get
    for attempt in range(max_retries):
        response = getter(f"{BASE_URL}/{path}", headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            return data
        if response.status_code in RETRY_STATUS and attempt < max_retries - 1:
            time.sleep(backoff_base * (2 ** attempt))
            continue
        response.raise_for_status()
    raise RuntimeError(f"API-Football no respondio 200 tras {max_retries} intentos ({path}).")


def is_cached(path: str, params: dict | None = None) -> bool:
    """True si la respuesta a (path, params) ya esta en cache (no costaria red).

    Lo usa la ingesta para respetar el presupuesto: solo cuenta como gasto un
    fixture cuyas stats aun no estan en disco.
    """
    return _cache_path(path, params or {}).exists()


# ---------------------------------------------------------------------------
# Parsing (puro, sin red)
# ---------------------------------------------------------------------------
def parse_fixture_statistics(payload: dict) -> pd.DataFrame:
    """Convierte /fixtures/statistics en una fila por equipo con las stats clave."""
    rows = []
    for entry in payload.get("response", []):
        stats = {STAT_TYPES[s["type"]]: s.get("value")
                 for s in entry.get("statistics", []) if s["type"] in STAT_TYPES}
        rows.append({"team": entry["team"]["name"], **stats})
    df = pd.DataFrame(rows)
    # Los valores pueden venir como None (sin dato) o como texto; a numerico.
    for col in STAT_TYPES.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_fixtures(payload: dict) -> pd.DataFrame:
    """Convierte /fixtures en una tabla de partidos con su arbitro designado."""
    rows = []
    for item in payload.get("response", []):
        fixture = item["fixture"]
        teams = item["teams"]
        rows.append({
            "fixture_id": fixture["id"],
            "date": fixture.get("date"),
            "referee": fixture.get("referee"),
            "status": fixture.get("status", {}).get("short"),
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
        })
    return pd.DataFrame(rows)


def referee_card_averages(fixtures_with_cards: pd.DataFrame) -> pd.DataFrame:
    """Promedio de tarjetas por partido de cada arbitro.

    fixtures_with_cards : DataFrame con columnas 'referee' y 'total_cards'.
    El factor arbitro del submodelo de tarjetas (Fase 2) se construye sobre esto.
    """
    valid = fixtures_with_cards.dropna(subset=["referee"])
    grouped = valid.groupby("referee")["total_cards"].agg(["mean", "count"]).reset_index()
    return grouped.rename(columns={"mean": "cards_per_match", "count": "matches"})


# ---------------------------------------------------------------------------
# Fetchers de alto nivel (red, cacheados)
# ---------------------------------------------------------------------------
def fixture_statistics(fixture_id: int, **kwargs) -> pd.DataFrame:
    """Estadisticas de un partido (corners, remates, tarjetas, faltas)."""
    return parse_fixture_statistics(get("fixtures/statistics", {"fixture": fixture_id}, **kwargs))


def fixtures(*, league: int, season: int, **kwargs) -> pd.DataFrame:
    """Partidos de una liga y temporada (con su arbitro)."""
    return parse_fixtures(get("fixtures", {"league": league, "season": season}, **kwargs))
