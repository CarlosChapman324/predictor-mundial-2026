"""Predicciones de los mercados de Capa 2 por partido del Mundial 2026.

Combina las tasas historicas por equipo (de API-Football, Fase 1) con las
fuerzas del modelo de goles para estimar corners, remates, remates a puerta y
tarjetas de cada partido, y derivar sus over/under. TODO confianza baja.

Las designaciones de arbitros del 2026 no estan en el plan free, asi que las
tarjetas usan el promedio general de arbitros como fallback (documentado en la UI).

    uv run python -m scripts.build_props
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model import extras, props

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

CORNERS_LINE = 9.5
CARDS_LINE = 4.5
# Nombres del dataset (modelo) -> nombres de API-Football (datos de stats).
TEAM_ALIASES = {"United States": "USA", "Turkey": "Türkiye", "Czech Republic": "Czechia"}


def main() -> None:
    stats = pd.read_parquet(PROCESSED_DIR / "capa2_team_match_stats.parquet")
    refs = pd.read_parquet(PROCESSED_DIR / "capa2_referee_cards.parquet")
    market = pd.read_parquet(PROCESSED_DIR / "match_markets.parquet")

    rates = props.team_rates(stats).set_index("team")
    league = {
        "corners": float(rates["corners_for"].mean()),
        "shots": float(rates["shots_for"].mean()),
        "cards": float(rates["cards_for"].mean()),
        "reds": float(rates["reds_for"].mean()),
        "sot_ratio": float(rates["sot_for"].sum() / rates["shots_for"].sum()),
    }
    league_referee_avg = float((refs["cards_per_match"] * refs["matches"]).sum() / refs["matches"].sum())
    avg_lambda = float(pd.concat([market["lambda_home"], market["lambda_away"]]).mean())

    def rate(team, column, default):
        name = TEAM_ALIASES.get(team, team)
        return float(rates.loc[name, column]) if name in rates.index else default

    def has_rates(team):
        return TEAM_ALIASES.get(team, team) in rates.index

    rows, missing = [], set()
    for r in market.itertuples(index=False):
        for team in (r.home_team, r.away_team):
            if not has_rates(team):
                missing.add(team)
        sm_h = props.strength_multiplier(r.lambda_home, avg_lambda)
        sm_a = props.strength_multiplier(r.lambda_away, avg_lambda)

        corners = (props.expected_count(rate(r.home_team, "corners_for", league["corners"]),
                                        rate(r.away_team, "corners_against", league["corners"]),
                                        league["corners"], sm_h)
                   + props.expected_count(rate(r.away_team, "corners_for", league["corners"]),
                                          rate(r.home_team, "corners_against", league["corners"]),
                                          league["corners"], sm_a))
        shots = (props.expected_count(rate(r.home_team, "shots_for", league["shots"]),
                                      rate(r.away_team, "shots_against", league["shots"]),
                                      league["shots"], sm_h)
                 + props.expected_count(rate(r.away_team, "shots_for", league["shots"]),
                                        rate(r.home_team, "shots_against", league["shots"]),
                                        league["shots"], sm_a))
        cards = props.expected_cards(rate(r.home_team, "cards_for", league["cards"]),
                                     rate(r.away_team, "cards_for", league["cards"]),
                                     None, league_referee_avg)
        reds = rate(r.home_team, "reds_for", league["reds"]) + rate(r.away_team, "reds_for", league["reds"])

        rows.append({
            "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
            "corners_total": round(corners, 2),
            "corners_over_9_5": extras.over_under_count(corners, CORNERS_LINE)["over"],
            "shots_total": round(shots, 2),
            "sot_total": round(shots * league["sot_ratio"], 2),
            "cards_total": round(cards, 2),
            "cards_over_4_5": extras.over_under_count(cards, CARDS_LINE)["over"],
            "red_card_prob": extras.red_card_probability(reds),
            "confidence": "baja",
        })

    out = pd.DataFrame(rows)
    out.to_parquet(PROCESSED_DIR / "props_predictions.parquet", index=False)

    teams_2026 = set(market["home_team"]) | set(market["away_team"])
    print(f"Predicciones de props para {len(out)} partidos (confianza baja).")
    print(f"Equipos con tasas reales: {len(teams_2026) - len(missing)}/{len(teams_2026)} "
          f"(el resto usa el promedio de la liga).")
    print(f"Promedio de arbitros (fallback de tarjetas): {league_referee_avg:.2f} tarjetas/partido")
    if missing:
        print("Sin tasas (fallback liga):", ", ".join(sorted(missing)))
    print("\nEjemplos:")
    cols = ["home_team", "away_team", "corners_total", "shots_total", "cards_total", "cards_over_4_5", "red_card_prob"]
    print(out[cols].head(6).to_string(index=False))
    print("\nListo. Props en data/processed/props_predictions.parquet (Capa 2, confianza baja).")


if __name__ == "__main__":
    main()
