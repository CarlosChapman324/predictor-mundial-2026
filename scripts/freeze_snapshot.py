"""Congela un snapshot de los datos que necesita el dashboard, para el deploy.

Los archivos de data/processed/ se generan en local y no se versionan (son
regenerables). Pero el dashboard desplegado en Streamlit Community Cloud no
corre los scripts de construccion, asi que necesita los datos ya calculados.
Este script copia solo lo que la app lee a data/snapshot/, que SI se versiona.

    uv run python -m scripts.freeze_snapshot

Correr antes de desplegar (o tras un recalculo importante que quieras publicar).
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SNAPSHOT = ROOT / "data" / "snapshot"

# Solo lo que consume app/data.py (se excluyen los pesados que la app no usa,
# como elo_history.parquet y matches.parquet).
SNAPSHOT_FILES = [
    "simulation_probabilities.parquet",
    "simulation_meta.json",
    "match_markets.parquet",
    "match_scorers.parquet",
    "golden_boot.parquet",
    "elo_current.parquet",
    "goal_model_strengths.parquet",
    "goal_model_params.json",
    "backtest_summary.parquet",
    "backtest_calibration.parquet",
    "market_comparison.parquet",
    "predictions_history.parquet",
    "props_predictions.parquet",
    "value_analysis.parquet",
    "predicted_vs_actual.parquet",
    "predicted_vs_actual_meta.json",
    "group_standings.parquet",
    "knockout_results.parquet",
]


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in SNAPSHOT_FILES:
        src = PROCESSED / name
        if src.exists():
            shutil.copy2(src, SNAPSHOT / name)
            copied.append(name)
        else:
            missing.append(name)

    total_kb = sum((SNAPSHOT / n).stat().st_size for n in copied) / 1024
    print(f"Snapshot congelado en data/snapshot/ ({len(copied)} archivos, {total_kb:.0f} KB).")
    if missing:
        print("Faltan (corre el build correspondiente si los quieres en el deploy):")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
