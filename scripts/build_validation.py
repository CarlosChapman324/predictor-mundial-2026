"""Corre el backtesting contra torneos pasados y guarda las metricas.

    uv run python -m scripts.build_validation

Entrena el modelo con corte temporal estricto antes de cada torneo de prueba,
predice sus partidos y compara contra los baselines (uniforme y Elo). Guarda:
  - backtest_summary.parquet     : RPS, log-loss y Brier por torneo y predictor
  - backtest_calibration.parquet : tabla de calibracion del modelo
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from validation import backtest

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    print("Backtesting contra Mundiales y Eurocopas (entrenando solo con el pasado)...\n")
    summary, calibration, _ = backtest.run_backtest(matches)

    summary.to_parquet(PROCESSED_DIR / "backtest_summary.parquet", index=False)
    calibration.to_parquet(PROCESSED_DIR / "backtest_calibration.parquet", index=False)
    (PROCESSED_DIR / "backtest_meta.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )

    # Tabla comparativa de RPS (menor es mejor).
    rps = summary.pivot(index="tournament", columns="predictor", values="rps")
    order = ["Mundial 2014", "Mundial 2018", "Mundial 2022", "Euro 2016", "Euro 2020", "Euro 2024", "Todos"]
    rps = rps.reindex([t for t in order if t in rps.index])
    n_by_t = summary[summary["predictor"] == "model"].set_index("tournament")["n"]

    print("RPS por torneo (menor es mejor):")
    print(f"  {'Torneo':<14}{'n':>5}{'Modelo':>9}{'Uniforme':>10}{'Elo':>8}{'Skill vs unif':>15}")
    for t in rps.index:
        m, u, e = rps.loc[t, "model"], rps.loc[t, "uniform"], rps.loc[t, "elo"]
        skill = 1.0 - m / u  # >0 significa que el modelo le gana al azar
        n = int(n_by_t.get(t, 0))
        print(f"  {t:<14}{n:>5}{m:>9.4f}{u:>10.4f}{e:>8.4f}{skill:>14.1%}")

    overall = summary[summary["tournament"] == "Todos"].set_index("predictor")
    print("\nGlobal (todos los partidos de prueba):")
    for predictor in ["model", "elo", "uniform"]:
        row = overall.loc[predictor]
        print(f"  {predictor:<9} RPS {row['rps']:.4f} | log-loss {row['log_loss']:.4f} | Brier {row['brier']:.4f}")

    print("\nCalibracion del modelo (probabilidad predicha vs frecuencia real):")
    print(f"  {'Predicha':>10}{'Observada':>11}{'n':>7}")
    for _, r in calibration.iterrows():
        print(f"  {r['mean_predicted']:>9.1%}{r['observed_frequency']:>11.1%}{int(r['count']):>7}")

    print("\nListo. Metricas en data/processed/backtest_*.parquet")


if __name__ == "__main__":
    main()
