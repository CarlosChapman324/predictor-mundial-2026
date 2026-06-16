"""Analisis de valor sobre los mercados con cuotas del Mundial 2026.

Calcula EV y la "mejor apuesta del partido" a partir de la comparacion modelo vs
mercado (1X2), y guarda value_analysis.parquet. Si hay partidos con cuotas Y
resultado, corre el backtest de la estrategia de valor frente a los baselines.

Nota honesta: un backtest sobre torneos pasados necesitaria cuotas historicas de
selecciones (no gratuitas); el motor (value.backtest) queda listo y testeado, y
se alimenta de los partidos del propio Mundial a medida que se juegan.

    uv run python -m scripts.build_value
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from value import backtest as value_backtest
from value import ev as value_ev

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    comparison_path = PROCESSED_DIR / "market_comparison.parquet"
    if not comparison_path.exists():
        print("Falta market_comparison.parquet. Corre antes: uv run python -m scripts.build_market")
        return

    comparison = pd.read_parquet(comparison_path)
    annotated = value_ev.annotate_value(comparison)
    annotated.to_parquet(PROCESSED_DIR / "value_analysis.parquet", index=False)

    n_value = int(annotated["has_value"].sum())
    media = annotated[annotated["has_value"] & (annotated["confidence"] == "media")]
    print(f"Partidos con cuotas analizados: {len(annotated)}")
    print(f"Margen medio de las casas: {comparison['overround'].mean():.1%}")
    print(f"Partidos con EV positivo: {n_value}/{len(annotated)} "
          f"({len(media)} de confianza media; el resto son longshots de confianza baja)")
    print("Que el modelo vea 'valor' casi en todos lados es la senal de un mercado")
    print("eficiente frente a un modelo menos extremo: NO es un edge real.")

    print("\nMayores EV (casi todos baja confianza: el modelo sobrevalora a los debiles):")
    top = annotated.sort_values("best_ev", ascending=False).head(8)
    for r in top.itertuples(index=False):
        odds = getattr(r, f"odds_{r.best_bet}")
        print(f"  {r.home_team:<16} vs {r.away_team:<16} {r.best_bet:>5}  "
              f"EV {r.best_ev:+7.1%}  cuota {odds:6.2f}  [{r.confidence}]")

    if not media.empty:
        print("\nUnicas 'mejores apuestas' de confianza media (mid-range, EV moderado):")
        for r in media.sort_values("best_ev", ascending=False).head(6).itertuples(index=False):
            print(f"  {r.home_team:<16} vs {r.away_team:<16} {r.best_bet:>5}  EV {r.best_ev:+6.1%}")

    # Backtest sobre lo que ya tenga cuotas Y resultado.
    if "outcome" in annotated.columns and annotated["outcome"].notna().any():
        result = value_backtest.backtest_strategy(annotated)
        v, fav = result["value"], result["favorite"]
        print(f"\nBacktest sobre {result['n_settled']} partidos liquidados:")
        print(f"  Estrategia de valor: {v['n_bets']} apuestas, ROI {v['roi']:+.1%}")
        print(f"  Baseline favorito:   ROI {fav['roi']:+.1%}")
        print("  (muestra minima; se acumula a medida que avanza el torneo)")
    else:
        print("\nAun no hay partidos con cuotas Y resultado para liquidar la estrategia")
        print("(las cuotas son de partidos por jugar). El motor de backtest esta listo y testeado.")

    print("\nNota: estudio de eficiencia de mercado, no un sistema de apuestas.")
    print("Listo. Analisis en data/processed/value_analysis.parquet")


if __name__ == "__main__":
    main()
