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

from data import fixture, ingest
from model import elo

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "reference"


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
