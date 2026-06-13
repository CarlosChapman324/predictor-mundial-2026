"""Smoke test de la Fase 0: la estructura de paquetes importa y los esqueletos
de fases futuras existen con la firma esperada. No prueba logica todavia.
"""

import importlib

import pytest

PACKAGES = [
    "data",
    "model",
    "model.elo",
    "model.goals",
    "model.markets",
    "tournament",
    "tournament.format2026",
    "tournament.montecarlo",
    "validation",
    "validation.metrics",
    "market",
    "market.odds",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_paquete_importa(name):
    importlib.import_module(name)


def test_los_esqueletos_de_fases_futuras_no_estan_implementados():
    from model import goals

    with pytest.raises(NotImplementedError):
        goals.score_matrix(1.4, 1.1, rho=0.0)
