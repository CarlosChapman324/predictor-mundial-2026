"""Construye todos los datos de la Capa 1 y los guarda en disco.

Pipeline:
    descargar historico -> normalizar -> matches.parquet
    matches -> Elo -> elo_history.parquet + elo_current.parquet
    historico + grupos + sedes -> calendario -> fixture_group_stage.parquet
    validar invariantes del formato 2026
    escribir _meta.json con timestamp

Uso:
    uv run python -m scripts.build_data            # descarga datos frescos
    uv run python -m scripts.build_data --no-download   # reusa el CSV cacheado

Es un script de orquestacion: aqui SI se puede tocar la red y el reloj. La
matematica (Elo) y la logica viven en los modulos puros, que este script llama.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import fixture, ingest
from model import elo
from tournament import format2026

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "reference"


def build_group_standings(schedule, groups, elo_current):
    """Tabla REAL de posiciones por grupo desde los resultados jugados.

    Aplica los mismos desempates 2026 del motor (head-to-head antes que la
    diferencia global, Elo como respaldo). 'qualified' marca al 1o y al 2o y,
    cuando la fase esta completa, tambien a los 8 mejores terceros.
    """
    final_key = dict(zip(elo_current["team"], elo_current["rating"]))
    played = schedule[schedule["played"]]
    complete = bool(schedule["played"].all())
    rows, thirds = [], []
    for group, block in groups.groupby("group"):
        teams_g = block["team"].tolist()
        results = [
            (r.home_team, int(r.home_score), r.away_team, int(r.away_score))
            for r in played[played["group"] == group].itertuples(index=False)
        ]
        ranked, overall = format2026.rank_group(teams_g, results, final_key=final_key)
        for position, team in enumerate(ranked, start=1):
            s = overall[team]
            rows.append({
                "group": group, "position": position, "team": team,
                "played": s["played"], "points": s["points"],
                "gf": s["gf"], "ga": s["ga"], "gd": s["gd"],
                "qualified": position <= 2,
            })
            if position == 3:
                thirds.append({"group": group, "team": team, "stats": s})
    standings = pd.DataFrame(rows)
    if complete and len(thirds) == 12:
        best_teams = {t["team"] for t in format2026.select_best_thirds(thirds, final_key=final_key)}
        third_mask = standings["team"].isin(best_teams) & (standings["position"] == 3)
        standings.loc[third_mask, "qualified"] = True
    return standings.sort_values(["group", "position"]).reset_index(drop=True)


def main(download: bool = True) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Historico ---------------------------------------------------------
    raw_csv = RAW_DIR / "results.csv"
    if download or not raw_csv.exists():
        print("Descargando historico de partidos...")
        raw_csv = ingest.download_results(RAW_DIR)
    else:
        print(f"Reusando historico cacheado: {raw_csv.relative_to(ROOT)}")

    raw = ingest.load_raw(raw_csv)
    matches = ingest.normalize_results(raw)
    ingest.save_parquet(matches, PROCESSED_DIR / "matches.parquet")
    print(f"  Partidos jugados normalizados: {len(matches):,} "
          f"({matches['date'].min().date()} -> {matches['date'].max().date()})")

    # 2. Elo ---------------------------------------------------------------
    print("Calculando Elo propio...")
    elo_history, elo_current = elo.compute_elo(matches)
    ingest.save_parquet(elo_history, PROCESSED_DIR / "elo_history.parquet")
    ingest.save_parquet(elo_current, PROCESSED_DIR / "elo_current.parquet")
    print(f"  Selecciones con rating: {len(elo_current):,}")

    # 3. Fixture 2026 ------------------------------------------------------
    print("Construyendo fixture 2026 (grupos, sedes, calendario)...")
    groups = fixture.load_groups(REFERENCE_DIR)
    venues = fixture.load_venues(REFERENCE_DIR)
    schedule = fixture.build_group_stage_schedule(raw, groups, venues)
    fixture.validate_fixture(groups, venues, schedule)
    ingest.save_parquet(schedule, PROCESSED_DIR / "fixture_group_stage.parquet")
    played = int(schedule["played"].sum())
    print(f"  Calendario fase de grupos: {len(schedule)} partidos "
          f"({played} jugados, {len(schedule) - played} pendientes)")

    # Eliminatoria ya jugada (capa viva del cuadro): cruces entre grupos distintos.
    shootouts_csv = RAW_DIR / "shootouts.csv"
    if download or not shootouts_csv.exists():
        shootouts_csv = ingest.download_shootouts(RAW_DIR)
    shootouts = ingest.load_raw(shootouts_csv)
    knockout = fixture.knockout_results(raw, groups, shootouts)
    ingest.save_parquet(knockout, PROCESSED_DIR / "knockout_results.parquet")
    print(f"  Eliminatoria jugada: {len(knockout)} cruces fijados")

    # Tabla real de posiciones por grupo (capa viva de la fase de grupos).
    standings = build_group_standings(schedule, groups, elo_current)
    ingest.save_parquet(standings, PROCESSED_DIR / "group_standings.parquet")
    qualified = int(standings["qualified"].sum())
    print(f"  Posiciones reales por grupo guardadas ({qualified} clasificados marcados)")

    # 4. Metadatos con timestamp ------------------------------------------
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": ingest.RESULTS_URL,
        "n_matches": int(len(matches)),
        "date_range": [str(matches["date"].min().date()), str(matches["date"].max().date())],
        "n_teams_rated": int(len(elo_current)),
        "fixture_matches": int(len(schedule)),
        "fixture_played": played,
    }
    (PROCESSED_DIR / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 5. Resumen legible ---------------------------------------------------
    print("\nTop 15 del Elo propio (rating actual):")
    top = elo_current.head(15).reset_index(drop=True)
    for i, row in top.iterrows():
        print(f"  {i + 1:>2}. {row['team']:<22} {row['rating']:>7.1f}")
    print("\nListo. Datos de la Capa 1 en data/processed/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye los datos de la Capa 1.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Reusa data/raw/results.csv en vez de descargar.",
    )
    args = parser.parse_args()
    main(download=not args.no_download)
