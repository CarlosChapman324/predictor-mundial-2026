"""Tests de la capa viva: guardado del historico de predicciones. Sin red."""

import pandas as pd
import pytest

from scripts import refresh


def _probs(champion):
    return pd.DataFrame({"team": ["A", "B"], "champion": champion, "qualify": [0.9, 0.8]})


def test_append_snapshot_acumula_y_reemplaza_misma_foto(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "HISTORY_PATH", tmp_path / "hist.parquet")

    refresh.append_snapshot(_probs([0.6, 0.4]), 0, "t0", "Sin condicionar")
    hist = refresh.append_snapshot(_probs([0.5, 0.5]), 4, "t1", "4 jugados")
    # Dos fotos distintas conviven.
    assert set(hist["matches_played"]) == {0, 4}

    # Recalcular la MISMA foto (4 jugados) la reemplaza, no la duplica.
    hist = refresh.append_snapshot(_probs([0.55, 0.45]), 4, "t2", "4 jugados")
    assert (hist["matches_played"] == 4).sum() == 2  # solo 2 equipos, sin duplicar
    a_value = hist[(hist["matches_played"] == 4) & (hist["team"] == "A")]["champion"].iloc[0]
    assert a_value == pytest.approx(0.55)
    # La foto en 0 (sin condicionar) sigue intacta.
    assert set(hist["matches_played"]) == {0, 4}
