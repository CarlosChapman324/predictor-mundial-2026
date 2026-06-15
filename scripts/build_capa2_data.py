"""Ingesta de datos de la Capa 2 desde API-Football (acceso directo de API-Sports).

El plan free solo da acceso a las temporadas 2022-2024 (no al 2026). Por eso se
toman las stats reales de torneos recientes de selecciones (Mundial 2022, Euro
2024, Copa America 2024) como base para estimar las tasas por equipo (corners,
remates, faltas, tarjetas) y el historial de tarjetas por arbitro. Las
designaciones de arbitros del Mundial 2026 NO son accesibles en free; el
submodelo de tarjetas (Fase 2) usara como fallback el promedio general hasta
conocerlas.

Respeta el limite de 100 requests/dia: cada respuesta se cachea en disco y, por
corrida, solo se descargan hasta --budget estadisticas de partidos nuevas. El
resto se completa en corridas siguientes (cron diario) sin volver a pedir lo
cacheado.

    uv run python -m scripts.build_capa2_data [--budget 80]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from data import apifootball

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Estados que cuentan como partido terminado (incluye prorroga y penales).
FINISHED = {"FT", "AET", "PEN"}

# Torneos de selecciones accesibles en el plan free (2022-2024), en orden de
# prioridad: lo mas valioso primero, para que el presupuesto cubra eso antes.
SOURCES = [
    {"league": 1, "season": 2022, "name": "Mundial 2022"},
    {"league": 4, "season": 2024, "name": "Euro 2024"},
    {"league": 9, "season": 2024, "name": "Copa America 2024"},
]


def _collect_fixtures() -> pd.DataFrame:
    """Trae los partidos jugados de cada torneo fuente (cacheado, 1 request c/u)."""
    frames = []
    for src in SOURCES:
        try:
            payload = apifootball.get("fixtures", {"league": src["league"], "season": src["season"]})
        except Exception as exc:  # noqa: BLE001
            print(f"  {src['name']}: no accesible ({exc}); se omite.")
            continue
        if payload.get("results", 0) == 0:
            print(f"  {src['name']}: sin datos ({payload.get('errors')}); se omite.")
            continue
        df = apifootball.parse_fixtures(payload)
        df = df[df["status"].isin(FINISHED)].copy()
        df["tournament"] = src["name"]
        frames.append(df)
        print(f"  {src['name']}: {len(df)} partidos jugados")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(budget: int, sleep_s: float) -> None:
    apifootball.load_env()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Trayendo fixtures de los torneos fuente (2022-2024)...")
    fixtures = _collect_fixtures()
    if fixtures.empty:
        print("No se pudo traer ningun torneo. Revisa la clave y la suscripcion.")
        return

    # Stats por partido, gastando como mucho `budget` requests nuevas.
    print(f"\nDescargando estadisticas por partido (presupuesto de red: {budget})...")
    used, skipped = 0, 0
    stat_rows = []
    total_cards = {}
    for row in fixtures.itertuples(index=False):
        fid = int(row.fixture_id)
        cached = apifootball.is_cached("fixtures/statistics", {"fixture": fid})
        if not cached and used >= budget:
            skipped += 1
            continue
        sdf = apifootball.fixture_statistics(fid)
        if not cached:
            used += 1
            time.sleep(sleep_s)  # respeta el limite por minuto del plan free
        if sdf.empty:
            continue
        for s in sdf.itertuples(index=False):
            stat_rows.append({
                "fixture_id": fid, "date": row.date, "tournament": row.tournament, "team": s.team,
                "corners": getattr(s, "corners", None), "shots": getattr(s, "shots", None),
                "shots_on_target": getattr(s, "shots_on_target", None), "fouls": getattr(s, "fouls", None),
                "yellow_cards": getattr(s, "yellow_cards", None), "red_cards": getattr(s, "red_cards", None),
            })
        total_cards[fid] = float(sdf[["yellow_cards", "red_cards"]].fillna(0).to_numpy().sum())

    team_stats = pd.DataFrame(stat_rows)
    team_stats.to_parquet(PROCESSED_DIR / "capa2_team_match_stats.parquet", index=False)
    fixtures.to_parquet(PROCESSED_DIR / "capa2_fixtures.parquet", index=False)

    # Historial de tarjetas por arbitro (solo de los partidos con stats ya bajadas).
    fx = fixtures.copy()
    fx["total_cards"] = fx["fixture_id"].map(total_cards)
    referees = apifootball.referee_card_averages(fx.dropna(subset=["total_cards"]))
    referees = referees.sort_values("cards_per_match", ascending=False).reset_index(drop=True)
    referees.to_parquet(PROCESSED_DIR / "capa2_referee_cards.parquet", index=False)

    n_with_stats = team_stats["fixture_id"].nunique() if not team_stats.empty else 0
    print(f"\nRequests de red usadas esta corrida: {used} (saltadas por presupuesto: {skipped})")
    print(f"Partidos con estadisticas en disco: {n_with_stats} | arbitros con historial: {len(referees)}")
    print("\nTop arbitros por tarjetas/partido (datos accesibles, base del factor arbitro):")
    print(f"  {'Arbitro':<22}{'Tarj/partido':>13}{'Partidos':>10}")
    for r in referees.head(8).itertuples(index=False):
        print(f"  {r.referee:<22}{r.cards_per_match:>13.2f}{r.matches:>10}")
    if skipped:
        print(f"\nQuedan {skipped} partidos sin stats; corre de nuevo (o deja al cron) para completarlos.")

    # Estado real de la cuota (1 request; nunca cacheado, es un contador en vivo).
    status = apifootball.get("status", use_cache=False).get("response", {}).get("requests", {})
    print(f"\nCuota API hoy: {status.get('current')}/{status.get('limit_day')}")
    print("Listo. Datos de Capa 2 en data/processed/ (confianza baja).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de datos de la Capa 2 (API-Football).")
    parser.add_argument("--budget", type=int, default=80,
                        help="Maximo de estadisticas de partido NUEVAS a descargar esta corrida.")
    parser.add_argument("--sleep", type=float, default=6.5,
                        help="Segundos entre llamadas de red (respeta el limite por minuto).")
    args = parser.parse_args()
    main(args.budget, args.sleep)
