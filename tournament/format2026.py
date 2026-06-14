"""Formato del Mundial 2026: tablas de grupo, desempates y siembra del cuadro.

Reglas clave (verificadas contra el reglamento 2026):
  - Avanzan el 1o y el 2o de cada grupo (24) mas los 8 mejores terceros (32).
  - Desempate DENTRO de un grupo, en orden: puntos totales; luego, entre los
    equipos empatados, enfrentamiento directo (puntos, diferencia y goles ENTRE
    ellos) ANTES que la diferencia de goles global; luego diferencia y goles
    globales; al final, criterio de respaldo (aqui usamos el Elo como proxy del
    ranking FIFA, y un desempate aleatorio como ultimo recurso).
  - Los 8 mejores terceros se eligen entre los 12 por criterios globales (puntos,
    diferencia, goles) y se asignan a sus 8 llaves de la Ronda de 32.

Todo es logica pura (sin red, sin pandas en el bucle caliente) para que el motor
Monte Carlo pueda invocarlo decenas de miles de veces.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Estadisticas de grupo
# ---------------------------------------------------------------------------
def accumulate_stats(results, teams, subset=None) -> dict[str, dict]:
    """Acumula puntos, goles y diferencia por equipo.

    results : lista de (equipo1, goles1, equipo2, goles2).
    teams   : equipos a incluir en la tabla.
    subset  : si se da, solo cuenta partidos entre equipos de ese conjunto
              (asi se calcula el enfrentamiento directo entre empatados).
    """
    stats = {t: {"points": 0, "gf": 0, "ga": 0, "played": 0} for t in teams}
    for t1, g1, t2, g2 in results:
        if subset is not None and (t1 not in subset or t2 not in subset):
            continue
        if t1 not in stats or t2 not in stats:
            continue
        stats[t1]["gf"] += g1
        stats[t1]["ga"] += g2
        stats[t2]["gf"] += g2
        stats[t2]["ga"] += g1
        stats[t1]["played"] += 1
        stats[t2]["played"] += 1
        if g1 > g2:
            stats[t1]["points"] += 3
        elif g2 > g1:
            stats[t2]["points"] += 3
        else:
            stats[t1]["points"] += 1
            stats[t2]["points"] += 1
    for t in stats:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]
    return stats


def _tiebreak_key(team, overall, h2h, final_key, rng):
    """Clave de ordenacion (mayor es mejor) para empates dentro de un grupo.

    Orden 2026: enfrentamiento directo (puntos, diferencia, goles) ANTES que la
    diferencia de goles global; luego goles globales; luego el proxy de ranking;
    y por ultimo un valor aleatorio como desempate final.
    """
    return (
        h2h[team]["points"],
        h2h[team]["gd"],
        h2h[team]["gf"],
        overall[team]["gd"],
        overall[team]["gf"],
        final_key.get(team, 0.0) if final_key else 0.0,
        rng.random() if rng is not None else 0.0,
    )


def rank_group(teams, results, *, final_key=None, rng=None):
    """Ordena los equipos de un grupo del 1o al 4o aplicando los desempates 2026.

    Devuelve (lista_ordenada, estadisticas_globales). Las estadisticas sirven
    luego para comparar a los terceros entre grupos.
    """
    overall = accumulate_stats(results, teams)
    by_points = sorted(teams, key=lambda t: overall[t]["points"], reverse=True)

    ranked = []
    i = 0
    while i < len(by_points):
        points = overall[by_points[i]]["points"]
        block = [t for t in by_points if overall[t]["points"] == points]
        if len(block) == 1:
            ranked.append(block[0])
        else:
            # Mini liga del enfrentamiento directo: solo partidos entre los empatados.
            h2h = accumulate_stats(results, block, subset=set(block))
            block.sort(key=lambda t: _tiebreak_key(t, overall, h2h, final_key, rng), reverse=True)
            ranked.extend(block)
        i += len(block)
    return ranked, overall


# ---------------------------------------------------------------------------
# Mejores terceros
# ---------------------------------------------------------------------------
def select_best_thirds(thirds, *, n=8, final_key=None, rng=None):
    """Elige los n mejores terceros por criterios globales.

    thirds : lista de dicts {group, team, stats} (stats con points, gd, gf).
    Devuelve la lista ordenada de los n mejores (con su grupo).
    """
    def key(item):
        s = item["stats"]
        return (
            s["points"], s["gd"], s["gf"],
            final_key.get(item["team"], 0.0) if final_key else 0.0,
            rng.random() if rng is not None else 0.0,
        )

    return sorted(thirds, key=key, reverse=True)[:n]


def assign_thirds_to_slots(qualified_thirds, third_slots):
    """Asigna cada tercero clasificado a una llave de la Ronda de 32.

    Unica restriccion (aproximacion del Anexo C oficial): un tercero no puede
    caer en la llave que enfrenta al ganador de su propio grupo. Con 8 terceros y
    8 llaves siempre existe una asignacion valida; se busca por backtracking.

    qualified_thirds : lista de {group, team} (8).
    third_slots      : lista de {match, faced_group} (8).
    Devuelve dict match -> team.
    """
    n = len(third_slots)
    used = [False] * n
    chosen = {}

    def backtrack(slot_i):
        if slot_i == n:
            return True
        faced = third_slots[slot_i]["faced_group"]
        for ti in range(n):
            third = qualified_thirds[ti]
            if not used[ti] and third["group"] != faced:
                used[ti] = True
                chosen[slot_i] = ti
                if backtrack(slot_i + 1):
                    return True
                used[ti] = False
        return False

    backtrack(0)
    return {
        third_slots[si]["match"]: qualified_thirds[ti]["team"]
        for si, ti in chosen.items()
    }


# ---------------------------------------------------------------------------
# Cuadro (datos de referencia)
# ---------------------------------------------------------------------------
def load_bracket(reference_dir: Path | str) -> dict:
    """Carga bracket_2026.json y precomputa los datos que usa el motor."""
    data = json.loads((Path(reference_dir) / "bracket_2026.json").read_text(encoding="utf-8"))

    # Para cada llave con un tercero, el grupo del ganador al que enfrenta.
    third_slots = []
    for match in data["round_of_32"]:
        slots = match["slots"]
        if any(s["type"] == "third" for s in slots):
            faced = next(s["group"] for s in slots if s["type"] != "third")
            third_slots.append({"match": match["match"], "faced_group": faced})
    data["third_slots"] = third_slots

    # Claves del arbol a enteros.
    data["tree"] = {int(k): v for k, v in data["tree"].items()}
    data["rounds"] = {k: list(v) for k, v in data["rounds"].items()}
    return data


def seed_round_of_32(ranked_by_group, stats_by_group, bracket, *, final_key=None, rng=None):
    """Construye los 16 enfrentamientos de la Ronda de 32.

    ranked_by_group : dict grupo -> lista ordenada de 4 equipos.
    stats_by_group  : dict grupo -> estadisticas globales por equipo.
    Devuelve (matchups, qualified_thirds) donde matchups es dict match -> (a, b).
    """
    winners = {g: ranked_by_group[g][0] for g in ranked_by_group}
    runners = {g: ranked_by_group[g][1] for g in ranked_by_group}

    thirds = [
        {"group": g, "team": ranked_by_group[g][2], "stats": stats_by_group[g][ranked_by_group[g][2]]}
        for g in ranked_by_group
    ]
    best = select_best_thirds(thirds, final_key=final_key, rng=rng)
    third_by_match = assign_thirds_to_slots(best, bracket["third_slots"])

    def resolve(slot, match_no):
        if slot["type"] == "winner":
            return winners[slot["group"]]
        if slot["type"] == "runner_up":
            return runners[slot["group"]]
        return third_by_match[match_no]  # 'third'

    matchups = {}
    for match in bracket["round_of_32"]:
        m = match["match"]
        a, b = match["slots"]
        matchups[m] = (resolve(a, m), resolve(b, m))
    return matchups, best
