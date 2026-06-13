"""Carga del fixture oficial del Mundial 2026: grupos, sedes y calendario.

Los grupos (A-L) y las sedes son datos curados a mano que viven en
data/reference/ (versionados en git). El calendario de la fase de grupos se
EXTRAE del propio historico: el dataset martj42 ya trae los 72 partidos de fase
de grupos del 2026 (con marcador NA los que faltan por jugar, lo que alimenta la
'capa viva' de la Fase 7).

Una validacion estricta comprueba que los datos curados y el calendario son
coherentes con el formato 2026 (12 grupos de 4, 48 equipos, 6 partidos por grupo).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Identificador de los partidos de fase de grupos del 2026 dentro del historico.
WORLD_CUP_TOURNAMENT = "FIFA World Cup"
WORLD_CUP_SEASON = "2026"

N_GROUPS = 12
TEAMS_PER_GROUP = 4
MATCHES_PER_GROUP = 6  # todos contra todos: C(4,2) = 6


def load_groups(reference_dir: Path | str) -> pd.DataFrame:
    """Lee groups.json y lo devuelve en formato largo: una fila por equipo.

    Columnas: group (A-L), seed_position (1-4), team.
    """
    data = json.loads((Path(reference_dir) / "groups.json").read_text(encoding="utf-8"))
    rows = []
    for group, teams in data["grupos"].items():
        for position, team in enumerate(teams, start=1):
            rows.append({"group": group, "seed_position": position, "team": team})
    return pd.DataFrame(rows)


def load_venues(reference_dir: Path | str) -> pd.DataFrame:
    """Lee venues.json: stadium, city, country, dataset_city."""
    data = json.loads((Path(reference_dir) / "venues.json").read_text(encoding="utf-8"))
    return pd.DataFrame(data["sedes"])


def team_to_group(groups: pd.DataFrame) -> dict[str, str]:
    """Diccionario equipo -> letra de grupo."""
    return dict(zip(groups["team"], groups["group"]))


def build_group_stage_schedule(
    raw: pd.DataFrame, groups: pd.DataFrame, venues: pd.DataFrame
) -> pd.DataFrame:
    """Construye el calendario de la fase de grupos a partir del historico crudo.

    Toma los partidos 'FIFA World Cup' de 2026 (jugados y por jugar), les asigna
    su grupo (ambos equipos comparten grupo) y su sede (uniendo por ciudad), y
    marca cuales ya se jugaron.
    """
    schedule = raw.loc[
        (raw["tournament"] == WORLD_CUP_TOURNAMENT)
        & (raw["date"].astype(str).str.startswith(WORLD_CUP_SEASON))
    ].copy()
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")

    mapping = team_to_group(groups)
    schedule["home_group"] = schedule["home_team"].map(mapping)
    schedule["away_group"] = schedule["away_team"].map(mapping)
    # En fase de grupos ambos equipos son del mismo grupo; ese es el grupo del partido.
    schedule["group"] = schedule["home_group"]

    venue_by_city = venues.set_index("dataset_city")[["stadium", "city", "country"]]
    schedule = schedule.merge(
        venue_by_city, left_on="city", right_index=True, how="left", suffixes=("", "_venue")
    )

    schedule["played"] = schedule["home_score"].notna() & schedule["away_score"].notna()

    columns = [
        "date", "group", "home_team", "away_team",
        "home_score", "away_score", "played",
        "stadium", "city_venue", "country",
    ]
    schedule = schedule[columns].rename(columns={"city_venue": "city"})
    return schedule.sort_values(["date", "group"]).reset_index(drop=True)


def validate_fixture(
    groups: pd.DataFrame, venues: pd.DataFrame, schedule: pd.DataFrame
) -> None:
    """Comprueba los invariantes del formato 2026. Lanza AssertionError si algo falla.

    Es la red de seguridad: si la fuente curada o el historico cambian de forma
    incoherente, el build falla en vez de propagar datos malos al modelo.
    """
    # Grupos
    assert groups["group"].nunique() == N_GROUPS, "Deben ser 12 grupos (A-L)"
    sizes = groups.groupby("group")["team"].size()
    assert (sizes == TEAMS_PER_GROUP).all(), "Cada grupo debe tener 4 equipos"
    assert groups["team"].nunique() == N_GROUPS * TEAMS_PER_GROUP, "Deben ser 48 equipos unicos"
    assert not groups["team"].duplicated().any(), "Ningun equipo puede estar en dos grupos"

    # Sedes
    assert len(venues) == 16, "Deben ser 16 sedes"
    assert venues["dataset_city"].is_unique, "Cada sede debe mapear a una ciudad distinta"

    # Calendario
    assert len(schedule) == N_GROUPS * MATCHES_PER_GROUP, "Deben ser 72 partidos de fase de grupos"
    assert schedule["group"].notna().all(), "Todo partido debe tener grupo asignado"
    per_group = schedule.groupby("group").size()
    assert (per_group == MATCHES_PER_GROUP).all(), "Cada grupo debe tener 6 partidos"
    assert schedule["stadium"].notna().all(), "Toda sede debe resolverse (revisa dataset_city)"

    # Coherencia entre calendario y grupos: los equipos del calendario son los del sorteo.
    teams_schedule = set(schedule["home_team"]) | set(schedule["away_team"])
    teams_groups = set(groups["team"])
    assert teams_schedule == teams_groups, (
        "Los equipos del calendario no coinciden con los del sorteo: "
        f"{teams_schedule.symmetric_difference(teams_groups)}"
    )
