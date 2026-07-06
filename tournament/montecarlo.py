"""Motor Monte Carlo del torneo.

Simula el Mundial completo muchas veces. En cada simulacion:
  1. Cada partido de grupo se resuelve muestreando un marcador desde la matriz
     del modelo de goles (los partidos YA JUGADOS se fijan con su resultado real:
     esa es la base de la 'capa viva' de la Fase 7).
  2. Se ordenan los grupos con los desempates 2026 y se eligen los 8 mejores terceros.
  3. Se siembra y se resuelve el cuadro hasta el campeon; cada cruce se decide con
     la probabilidad de avance del modelo (gana en 90 min o, si hay empate, ~50%
     en penales).
Agregando las frecuencias se obtienen las probabilidades de ganar el grupo,
clasificar, llegar a cada ronda y ser campeon.

La localia solo se aplica a los anfitriones (USA, Canada, Mexico) en sus partidos
de grupo jugados en su pais; las eliminatorias se tratan como cancha neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.goals import FittedGoalModel
from tournament import format2026

# Anfitrion -> pais de la sede (para aplicar localia solo cuando juega en casa).
HOST_COUNTRY = {"Mexico": "Mexico", "United States": "United States", "Canada": "Canada"}

# Codigos de etapa alcanzada (la ronda mas lejana que jugo el equipo).
STAGE_R32, STAGE_R16, STAGE_QF, STAGE_SF, STAGE_FINAL, STAGE_CHAMPION = 1, 2, 3, 4, 5, 6


def _host_with_advantage(home_team, away_team, venue_country):
    """Devuelve el anfitrion que juega en su pais en este partido, o None."""
    if HOST_COUNTRY.get(home_team) == venue_country:
        return home_team
    if HOST_COUNTRY.get(away_team) == venue_country:
        return away_team
    return None


def _prepare_group_matches(fixture: pd.DataFrame, model: FittedGoalModel, max_goals: int,
                           ignore_played: bool = False):
    """Para cada partido de grupo decide quien tiene localia y precomputa su
    distribucion de marcadores (o fija el resultado si ya se jugo).

    ignore_played=True trata TODOS los partidos como pendientes (sirve para el
    pronostico 'pre-torneo', sin condicionar a lo que ya paso).
    """
    width = max_goals + 1
    prepared = []
    for row in fixture.itertuples(index=False):
        adv_team = _host_with_advantage(row.home_team, row.away_team, row.country)
        # team_a es el lado que se modela como local (el anfitrion si aplica).
        if adv_team == row.away_team:
            team_a, team_b, swap = row.away_team, row.home_team, True
        else:
            team_a, team_b, swap = row.home_team, row.away_team, False
        home_adv = adv_team is not None

        match = {"group": row.group, "team_a": team_a, "team_b": team_b}
        if bool(row.played) and not ignore_played:
            ga, gb = int(row.home_score), int(row.away_score)
            match["fixed"] = (gb, ga) if swap else (ga, gb)  # goles (team_a, team_b)
        else:
            matrix = model.score_matrix(team_a, team_b, home_advantage=home_adv, max_goals=max_goals)
            match["cum"] = np.cumsum(matrix.ravel())
            match["width"] = width
        prepared.append(match)
    return prepared


def _advance_probability(model: FittedGoalModel, teams, max_goals: int) -> dict:
    """p[a][b] = probabilidad de que a elimine a b en cancha neutral.

    Gana en tiempo reglamentario, o la mitad de la probabilidad de empate
    (los penales se modelan como un volado parejo).
    """
    p = {a: {} for a in teams}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            matrix = model.score_matrix(a, b, home_advantage=False, max_goals=max_goals)
            home_win = np.tril(matrix, -1).sum()
            draw = np.trace(matrix)
            p[a][b] = float(home_win + 0.5 * draw)
    return p


def _play_tournament(group_to_teams, group_results, padv, bracket, final_key, ko_randoms,
                     played_knockout=None):
    """Resuelve un torneo completo a partir de los resultados de grupo ya decididos.

    Devuelve (positions, qualified, stage, champion).
    """
    ranked_by_group, stats_by_group = {}, {}
    for g, teams in group_to_teams.items():
        ranked, stats = format2026.rank_group(teams, group_results[g], final_key=final_key)
        ranked_by_group[g] = ranked
        stats_by_group[g] = stats

    matchups, best_thirds = format2026.seed_round_of_32(
        ranked_by_group, stats_by_group, bracket, final_key=final_key
    )

    positions = {}
    for g, ranked in ranked_by_group.items():
        for pos, team in enumerate(ranked, start=1):
            positions[team] = pos
    qualified = set()
    for g, ranked in ranked_by_group.items():
        qualified.update({ranked[0], ranked[1]})
    qualified.update(t["team"] for t in best_thirds)

    stage = {team: STAGE_R32 for team in qualified}
    winner_of = {}
    k = 0  # indice en el vector de aleatorios de eliminatorias

    # Equipos ya eliminados en la realidad (perdieron un cruce jugado): aunque el
    # cuadro reconstruido los empareje distinto, nunca deben avanzar.
    eliminated = set()
    if played_knockout:
        for pair, winner in played_knockout.items():
            eliminated.update(t for t in pair if t != winner)

    def resolve(a, b):
        nonlocal k
        r = ko_randoms[k]
        k += 1
        if played_knockout:
            fixed = played_knockout.get(frozenset((a, b)))
            if fixed is not None:
                return fixed          # cruce ya jugado: se fija con el resultado real
            if a in eliminated and b not in eliminated:
                return b              # a ya quedo fuera en la realidad
            if b in eliminated and a not in eliminated:
                return a
        return a if r < padv[a][b] else b

    # Ronda de 32 (cruces sembrados).
    for m in bracket["rounds"]["round_of_32"]:
        a, b = matchups[m]
        w = resolve(a, b)
        winner_of[m] = w
        stage[w] = STAGE_R16
    # Rondas siguientes (los rivales salen del arbol).
    for round_name, next_stage in [
        ("round_of_16", STAGE_QF),
        ("quarter_finals", STAGE_SF),
        ("semi_finals", STAGE_FINAL),
        ("final", STAGE_CHAMPION),
    ]:
        for m in bracket["rounds"][round_name]:
            f1, f2 = bracket["tree"][m]
            w = resolve(winner_of[f1], winner_of[f2])
            winner_of[m] = w
            stage[w] = next_stage

    champion = winner_of[bracket["rounds"]["final"][0]]
    return positions, qualified, stage, champion


def simulate_tournament(fixture, model, bracket, *, final_key=None, max_goals=10, rng=None,
                        ignore_played=False, played_knockout=None):
    """Simula UN torneo. Util para tests y para inspeccionar una corrida."""
    rng = rng or np.random.default_rng()
    prepared = _prepare_group_matches(fixture, model, max_goals, ignore_played=ignore_played)
    teams = sorted(set(fixture["home_team"]) | set(fixture["away_team"]))
    padv = _advance_probability(model, teams, max_goals)

    group_to_teams, group_results = _group_structure(prepared)
    for match in prepared:
        if "fixed" in match:
            ga, gb = match["fixed"]
        else:
            idx = int(np.searchsorted(match["cum"], rng.random()))
            ga, gb = divmod(idx, match["width"])
        group_results[match["group"]].append((match["team_a"], ga, match["team_b"], gb))

    ko_randoms = rng.random(40)
    return _play_tournament(group_to_teams, group_results, padv, bracket, final_key, ko_randoms,
                            played_knockout)


def _group_structure(prepared):
    """Equipos por grupo y un contenedor vacio de resultados por grupo."""
    group_to_teams = {}
    for match in prepared:
        group_to_teams.setdefault(match["group"], set()).update({match["team_a"], match["team_b"]})
    group_to_teams = {g: sorted(ts) for g, ts in group_to_teams.items()}
    group_results = {g: [] for g in group_to_teams}
    return group_to_teams, group_results


def run_monte_carlo(
    fixture, model, bracket, *, n_sims=10_000, final_key=None, max_goals=10, seed=0,
    ignore_played=False, played_knockout=None,
) -> pd.DataFrame:
    """Corre n_sims torneos y agrega las probabilidades por seleccion.

    Devuelve un DataFrame con, por equipo: probabilidad de ganar el grupo, de
    quedar segundo, de clasificar como tercero, de clasificar (llegar a la Ronda
    de 32), de alcanzar cada ronda y de ser campeon.

    ignore_played=True ignora los resultados reales (pronostico pre-torneo).
    """
    rng = np.random.default_rng(seed)
    prepared = _prepare_group_matches(fixture, model, max_goals, ignore_played=ignore_played)
    teams = sorted(set(fixture["home_team"]) | set(fixture["away_team"]))
    padv = _advance_probability(model, teams, max_goals)
    group_to_teams, _ = _group_structure(prepared)

    # Pre-muestreo vectorizado de todos los marcadores de grupo (rapido).
    n_matches = len(prepared)
    goals_a = np.zeros((n_matches, n_sims), dtype=np.int16)
    goals_b = np.zeros((n_matches, n_sims), dtype=np.int16)
    for i, match in enumerate(prepared):
        if "fixed" in match:
            goals_a[i, :], goals_b[i, :] = match["fixed"]
        else:
            idx = np.searchsorted(match["cum"], rng.random(n_sims))
            goals_a[i, :], goals_b[i, :] = np.divmod(idx, match["width"])

    # Aleatorios de eliminatorias: una fila por simulacion.
    ko_randoms = rng.random((n_sims, 40))

    # Indices de partidos por grupo, para armar los resultados de cada simulacion.
    group_match_index = {g: [] for g in group_to_teams}
    for i, match in enumerate(prepared):
        group_match_index[match["group"]].append(i)

    counters = {
        t: {"champion": 0, "final": 0, "semifinal": 0, "quarterfinal": 0,
            "round_of_16": 0, "qualify": 0, "win_group": 0, "runner_up": 0, "third": 0}
        for t in teams
    }

    for s in range(n_sims):
        group_results = {g: [] for g in group_to_teams}
        for g, indices in group_match_index.items():
            for i in indices:
                m = prepared[i]
                group_results[g].append((m["team_a"], int(goals_a[i, s]), m["team_b"], int(goals_b[i, s])))

        positions, qualified, stage, champion = _play_tournament(
            group_to_teams, group_results, padv, bracket, final_key, ko_randoms[s], played_knockout
        )

        for team, pos in positions.items():
            if pos == 1:
                counters[team]["win_group"] += 1
            elif pos == 2:
                counters[team]["runner_up"] += 1
            elif pos == 3 and team in qualified:
                counters[team]["third"] += 1
        for team in qualified:
            c = counters[team]
            st = stage[team]
            c["qualify"] += 1
            if st >= STAGE_R16:
                c["round_of_16"] += 1
            if st >= STAGE_QF:
                c["quarterfinal"] += 1
            if st >= STAGE_SF:
                c["semifinal"] += 1
            if st >= STAGE_FINAL:
                c["final"] += 1
            if st >= STAGE_CHAMPION:
                c["champion"] += 1

    rows = []
    for team in teams:
        c = counters[team]
        rows.append({"team": team, **{k: v / n_sims for k, v in c.items()}})
    columns = ["team", "champion", "final", "semifinal", "quarterfinal",
               "round_of_16", "qualify", "win_group", "runner_up", "third"]
    return pd.DataFrame(rows)[columns].sort_values("champion", ascending=False).reset_index(drop=True)
