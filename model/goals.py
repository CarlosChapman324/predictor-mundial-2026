"""Modelo de goles: Poisson bivariado con correccion de Dixon-Coles.

Idea central
------------
Cada seleccion tiene una fuerza de ATAQUE (cuanto marca) y una de DEFENSA
(cuanto le marcan). Para un partido entre local i y visitante j:

    log(lambda_local)     = intercepto + localia + ataque_i + defensa_j
    log(lambda_visitante) = intercepto +           ataque_j + defensa_i

(la localia solo entra si el local de verdad juega en casa, no en cancha neutral).
Con esos dos lambda se construye una matriz de Poisson con la probabilidad de
cada marcador i-j. De esa matriz salen TODOS los mercados de la Capa 1.

Correccion de Dixon-Coles
-------------------------
El Poisson independiente asume que goles del local y del visitante no estan
correlacionados, y por eso SUBESTIMA los marcadores bajos y los empates. Dixon y
Coles (1997) multiplican las cuatro celdas bajas (0-0, 1-0, 0-1, 1-1) por un
factor tau que depende de un parametro de dependencia rho (tipicamente pequeno y
negativo). Es el ajuste que mas se nota en la calidad del modelo de futbol.

Estimacion
----------
Maxima verosimilitud sobre el historico, ponderando cada partido por:
  - decaimiento temporal (half-life): lo reciente pesa mas;
  - relevancia del torneo: un amistoso pesa menos que un partido oficial.
El ajuste es conjunto (ataque/defensa de todos, localia y rho a la vez) con
scipy y gradiente analitico. Una pequena regularizacion L2 sobre ataque/defensa
estabiliza a las selecciones con pocos partidos y fija la identificabilidad.

Todo es matematica pura: no toca la red. Recibe el DataFrame ya normalizado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# Peso por relevancia del torneo (multiplica al peso temporal). Un amistoso
# informa menos sobre la fuerza real que un partido oficial.
IMPORTANCE_WEIGHTS = {
    "world_cup": 1.0,
    "continental": 1.0,
    "confederations": 0.9,
    "qualifier": 0.9,
    "nations_league": 0.85,
    "other": 0.7,
    "friendly": 0.5,
}

# Cota de rho para mantener positivas las celdas ajustadas por Dixon-Coles.
RHO_BOUND = 0.2


# ---------------------------------------------------------------------------
# Matriz de marcadores (la fuente unica de todos los mercados)
# ---------------------------------------------------------------------------
def _dixon_coles_tau(matrix: np.ndarray, lambda_home: float, lambda_away: float, rho: float) -> np.ndarray:
    """Aplica el factor tau de Dixon-Coles a las cuatro celdas bajas."""
    adjusted = matrix.copy()
    adjusted[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    adjusted[1, 0] *= 1.0 + lambda_away * rho
    adjusted[0, 1] *= 1.0 + lambda_home * rho
    adjusted[1, 1] *= 1.0 - rho
    return adjusted


def score_matrix(
    lambda_home: float, lambda_away: float, rho: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Matriz (max_goals+1 x max_goals+1) con P(marcador i-j).

    Filas = goles del local, columnas = goles del visitante. Se construye el
    producto externo de dos Poisson independientes, se aplica el ajuste de
    Dixon-Coles en las celdas bajas y se renormaliza para que sume 1 (lo que
    tambien corrige la cola truncada del Poisson).
    """
    goals = np.arange(max_goals + 1)
    # P(k) de un Poisson, en log para estabilidad y luego exp.
    log_pmf_home = goals * np.log(lambda_home) - lambda_home - gammaln(goals + 1)
    log_pmf_away = goals * np.log(lambda_away) - lambda_away - gammaln(goals + 1)
    matrix = np.outer(np.exp(log_pmf_home), np.exp(log_pmf_away))

    if rho != 0.0:
        matrix = _dixon_coles_tau(matrix, lambda_home, lambda_away, rho)

    matrix = np.clip(matrix, 0.0, None)  # por si tau deja una celda apenas negativa
    return matrix / matrix.sum()


# ---------------------------------------------------------------------------
# Modelo ajustado
# ---------------------------------------------------------------------------
@dataclass
class FittedGoalModel:
    """Parametros estimados del modelo de goles, listos para predecir."""

    intercept: float
    home_advantage: float
    rho: float
    attack: dict[str, float]
    defense: dict[str, float]
    meta: dict = field(default_factory=dict)

    @property
    def teams(self) -> list[str]:
        return sorted(self.attack)

    def expected_goals(
        self, home: str, away: str, *, home_advantage: bool = False
    ) -> tuple[float, float]:
        """Goles esperados (lambda) de local y visitante para un enfrentamiento.

        Si una seleccion no se estimo (no aparece en el modelo), se trata como
        promedio: ataque y defensa cero.
        """
        a_home = self.attack.get(home, 0.0)
        a_away = self.attack.get(away, 0.0)
        d_home = self.defense.get(home, 0.0)
        d_away = self.defense.get(away, 0.0)
        adv = self.home_advantage if home_advantage else 0.0
        lambda_home = np.exp(self.intercept + adv + a_home + d_away)
        lambda_away = np.exp(self.intercept + a_away + d_home)
        return float(lambda_home), float(lambda_away)

    def score_matrix(
        self, home: str, away: str, *, home_advantage: bool = False, max_goals: int = 10
    ) -> np.ndarray:
        """Matriz de marcadores para un enfrentamiento concreto."""
        lambda_home, lambda_away = self.expected_goals(
            home, away, home_advantage=home_advantage
        )
        return score_matrix(lambda_home, lambda_away, self.rho, max_goals)


# ---------------------------------------------------------------------------
# Preparacion de datos y pesos
# ---------------------------------------------------------------------------
def _temporal_weight(dates: pd.Series, reference_date: pd.Timestamp, half_life_days: float | None) -> np.ndarray:
    """Decaimiento exponencial: peso = 0.5 ^ (antiguedad / half_life)."""
    if half_life_days is None:
        return np.ones(len(dates))
    age_days = (reference_date - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age_days / half_life_days)


def _prepare(
    matches: pd.DataFrame,
    *,
    half_life_days: float | None,
    max_age_years: float | None,
    min_matches: int,
    importance_weights: dict[str, float] | None,
    reference_date: pd.Timestamp | None,
):
    """Filtra la ventana, calcula pesos e indexa equipos para el ajuste."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date = reference_date or df["date"].max()

    if max_age_years is not None:
        cutoff = reference_date - pd.Timedelta(days=max_age_years * 365.25)
        df = df[df["date"] >= cutoff].copy()

    # Descartar selecciones con muy pocos partidos en la ventana: sus parametros
    # serian puro ruido. Nos quedamos con los partidos donde AMBOS superan el umbral.
    if min_matches > 0:
        appearances = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        frequent = set(appearances[appearances >= min_matches].index)
        df = df[df["home_team"].isin(frequent) & df["away_team"].isin(frequent)].copy()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    index = {team: i for i, team in enumerate(teams)}

    temporal = _temporal_weight(df["date"], reference_date, half_life_days)
    weights_table = importance_weights if importance_weights is not None else IMPORTANCE_WEIGHTS
    relevance = df["tournament_category"].map(weights_table).fillna(0.7).to_numpy(dtype=float)
    weights = temporal * relevance
    weights = weights / weights.mean()  # media 1, para que la regularizacion sea interpretable

    data = {
        "home_idx": df["home_team"].map(index).to_numpy(),
        "away_idx": df["away_team"].map(index).to_numpy(),
        "home_goals": df["home_score"].to_numpy(dtype=float),
        "away_goals": df["away_score"].to_numpy(dtype=float),
        "not_neutral": (~df["neutral"].to_numpy(dtype=bool)).astype(float),
        "weights": weights,
    }
    return data, teams, index, reference_date


def _negative_log_likelihood(theta, data, n_teams, l2):
    """NLL ponderada del modelo Dixon-Coles y su gradiente analitico.

    Layout de theta: [intercepto, localia, rho, ataque(n_teams), defensa(n_teams)].
    Devolver el gradiente exacto hace el ajuste rapido y estable.
    """
    intercept, home_adv, rho = theta[0], theta[1], theta[2]
    attack = theta[3 : 3 + n_teams]
    defense = theta[3 + n_teams : 3 + 2 * n_teams]

    hi, ai = data["home_idx"], data["away_idx"]
    x, y = data["home_goals"], data["away_goals"]
    not_neutral, w = data["not_neutral"], data["weights"]

    eta_home = intercept + home_adv * not_neutral + attack[hi] + defense[ai]
    eta_away = intercept + attack[ai] + defense[hi]
    lam = np.exp(eta_home)
    mu = np.exp(eta_away)

    # Parte Poisson (log-verosimilitud por partido).
    ll = x * eta_home - lam - gammaln(x + 1) + y * eta_away - mu - gammaln(y + 1)

    # Correccion de Dixon-Coles: factor tau y sus derivadas en las celdas bajas.
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)

    tau = np.ones_like(lam)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho
    tau = np.clip(tau, 1e-12, None)
    ll = ll + np.log(tau)

    nll = -np.sum(w * ll) + 0.5 * l2 * (attack @ attack + defense @ defense)

    # --- Gradiente ---------------------------------------------------------
    # Derivada de log(tau) respecto de lam, mu y rho (cero salvo en celdas bajas).
    dlogtau_dlam = np.zeros_like(lam)
    dlogtau_dmu = np.zeros_like(lam)
    dlogtau_drho = np.zeros_like(lam)

    dlogtau_dlam[m00] = -mu[m00] * rho / tau[m00]
    dlogtau_dmu[m00] = -lam[m00] * rho / tau[m00]
    dlogtau_drho[m00] = -lam[m00] * mu[m00] / tau[m00]

    dlogtau_dmu[m10] = rho / tau[m10]
    dlogtau_drho[m10] = mu[m10] / tau[m10]

    dlogtau_dlam[m01] = rho / tau[m01]
    dlogtau_drho[m01] = lam[m01] / tau[m01]

    dlogtau_drho[m11] = -1.0 / tau[m11]

    # Residuos por "canal": como d lambda / d eta = lambda, la regla de la cadena
    # mete el termino de tau junto al residuo de Poisson (x - lambda).
    res_home = (x - lam) + lam * dlogtau_dlam
    res_away = (y - mu) + mu * dlogtau_dmu

    grad = np.zeros_like(theta)
    grad[0] = -np.sum(w * (res_home + res_away))            # intercepto
    grad[1] = -np.sum(w * not_neutral * res_home)           # localia
    grad[2] = -np.sum(w * dlogtau_drho)                     # rho
    # ataque y defensa: dispersar los residuos a cada equipo.
    grad_attack = -(np.bincount(hi, w * res_home, n_teams) + np.bincount(ai, w * res_away, n_teams))
    grad_defense = -(np.bincount(ai, w * res_home, n_teams) + np.bincount(hi, w * res_away, n_teams))
    grad[3 : 3 + n_teams] = grad_attack + l2 * attack
    grad[3 + n_teams :] = grad_defense + l2 * defense

    return nll, grad


def fit_goal_model(
    matches: pd.DataFrame,
    *,
    half_life_days: float | None = 730.0,
    max_age_years: float | None = 12.0,
    min_matches: int = 10,
    importance_weights: dict[str, float] | None = None,
    l2: float = 1.0,
    reference_date: pd.Timestamp | None = None,
) -> FittedGoalModel:
    """Estima el modelo de goles por maxima verosimilitud ponderada.

    Parametros
    ----------
    matches : historico normalizado (date, home_team, away_team, home_score,
        away_score, neutral, tournament_category).
    half_life_days : vida media del decaimiento temporal. None = sin decaimiento.
    max_age_years : ventana de anos hacia atras a usar. None = todo el historico.
    min_matches : umbral minimo de partidos por seleccion en la ventana.
    importance_weights : peso por categoria de torneo. None usa el por defecto.
    l2 : fuerza de la regularizacion (shrinkage hacia el promedio).
    reference_date : "hoy" para el decaimiento. None = ultima fecha del historico.
    """
    data, teams, index, reference_date = _prepare(
        matches,
        half_life_days=half_life_days,
        max_age_years=max_age_years,
        min_matches=min_matches,
        importance_weights=importance_weights,
        reference_date=reference_date,
    )
    n_teams = len(teams)

    # Valores iniciales razonables: intercepto = log(promedio de goles por equipo).
    mean_goals = np.mean(np.concatenate([data["home_goals"], data["away_goals"]]))
    x0 = np.zeros(3 + 2 * n_teams)
    x0[0] = np.log(max(mean_goals, 0.1))
    x0[1] = 0.2    # localia inicial
    x0[2] = -0.05  # rho inicial (pequeno y negativo, como suele salir)

    bounds = [(None, None), (None, None), (-RHO_BOUND, RHO_BOUND)] + [(None, None)] * (2 * n_teams)

    result = minimize(
        _negative_log_likelihood,
        x0,
        args=(data, n_teams, l2),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500},
    )

    attack = {team: float(result.x[3 + i]) for i, team in enumerate(teams)}
    defense = {team: float(result.x[3 + n_teams + i]) for i, team in enumerate(teams)}
    meta = {
        "n_matches": int(len(data["weights"])),
        "n_teams": n_teams,
        "reference_date": str(pd.Timestamp(reference_date).date()),
        "half_life_days": half_life_days,
        "max_age_years": max_age_years,
        "min_matches": min_matches,
        "l2": l2,
        "converged": bool(result.success),
        "final_nll": float(result.fun),
    }
    return FittedGoalModel(
        intercept=float(result.x[0]),
        home_advantage=float(result.x[1]),
        rho=float(result.x[2]),
        attack=attack,
        defense=defense,
        meta=meta,
    )
