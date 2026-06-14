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


def test_los_modulos_clave_estan_implementados():
    # Ya no quedan esqueletos NotImplementedError en el nucleo: humo positivo.
    from market import odds

    probs = odds.implied_probabilities([2.0, 4.0, 4.0])
    assert abs(float(probs.sum()) - 1.0) < 1e-9
