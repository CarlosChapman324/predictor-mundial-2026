"""Construye los mercados de goleadores de la Capa 2 (experimental).

    uv run python -m scripts.build_scorers [--no-download]

Combina los goleadores recientes (goalscorers.csv) con el modelo de goles y la
simulacion para producir:
  - match_scorers.parquet : probabilidad de marcar de cada jugador por partido.
  - golden_boot.parquet   : proyeccion de goles en el torneo y prob. de Bota de Oro.

CONFIANZA BAJA: usa a los goleadores de los ultimos 4 anos como proxy del plantel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import scorers as scorers_data
from model import scorers

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
SHARE_WINDOW_YEARS = 4
TOP_SCORERS_PER_MATCH = 10
GOLDEN_BOOT_CONTENDERS = 80  # jugadores con opcion real, para la prob. de Bota de Oro


def main(download: bool = True) -> None:
    raw_csv = RAW_DIR / "goalscorers.csv"
    if download or not raw_csv.exists():
        print("Descargando goleadores internacionales...")
        raw_csv = scorers_data.download_goalscorers(RAW_DIR)
    goalscorers = pd.read_csv(raw_csv)

    since = pd.to_datetime(goalscorers["date"]).max() - pd.DateOffset(years=SHARE_WINDOW_YEARS)
    shares = scorers_data.player_shares(goalscorers, since=since)
    shares_by_team = {team: block[["scorer", "share"]] for team, block in shares.groupby("team")}
    print(f"Cuotas de gol calculadas desde {since.date()} ({shares['team'].nunique()} selecciones).")

    market = pd.read_parquet(PROCESSED_DIR / "match_markets.parquet")
    sim = pd.read_parquet(PROCESSED_DIR / "simulation_probabilities.parquet")

    # Goles esperados por partido (promedio del equipo) y partidos esperados.
    home = market[["home_team", "lambda_home"]].rename(columns={"home_team": "team", "lambda_home": "lam"})
    away = market[["away_team", "lambda_away"]].rename(columns={"away_team": "team", "lambda_away": "lam"})
    team_lambda = pd.concat([home, away]).groupby("team")["lam"].mean().to_dict()
    sim = sim.copy()
    sim["exp_matches"] = 3 + sim["qualify"] + sim["round_of_16"] + sim["quarterfinal"] + sim["semifinal"] + sim["final"]
    expected_matches = dict(zip(sim["team"], sim["exp_matches"]))

    empty = pd.DataFrame(columns=["scorer", "share"])

    # Goleadores por partido.
    rows = []
    for r in market.itertuples(index=False):
        table = scorers.match_scorers(
            r.home_team, r.lambda_home, shares_by_team.get(r.home_team, empty),
            r.away_team, r.lambda_away, shares_by_team.get(r.away_team, empty),
            top_n=TOP_SCORERS_PER_MATCH,
        )
        for t in table.itertuples(index=False):
            rows.append({"date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                         "player": t.player, "player_team": t.team, "anytime": t.anytime})
    pd.DataFrame(rows).to_parquet(PROCESSED_DIR / "match_scorers.parquet", index=False)

    # Bota de Oro: proyeccion y probabilidad.
    golden = scorers.golden_boot_projection(shares, team_lambda, expected_matches)
    contenders = golden.head(GOLDEN_BOOT_CONTENDERS).copy()
    contenders["win_prob"] = scorers.golden_boot_probabilities(contenders["expected_goals"].to_numpy())
    golden["win_prob"] = 0.0
    golden.loc[contenders.index, "win_prob"] = contenders["win_prob"].values
    golden.head(40).to_parquet(PROCESSED_DIR / "golden_boot.parquet", index=False)

    print("\nBota de Oro (proyeccion, experimental):")
    print(f"  {'Jugador':<22}{'Seleccion':<16}{'Goles esp.':>11}{'Prob.':>8}")
    for r in golden.head(12).itertuples(index=False):
        print(f"  {r.player:<22}{r.team:<16}{r.expected_goals:>11.2f}{r.win_prob:>8.1%}")
    print("\nListo. Mercados de goleadores en data/processed/ (Capa 2, confianza baja).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mercados de goleadores (Capa 2).")
    parser.add_argument("--no-download", action="store_true", help="Reusa goalscorers.csv cacheado.")
    args = parser.parse_args()
    main(download=not args.no_download)
