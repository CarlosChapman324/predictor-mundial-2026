"""Tests del cliente de API-Football (Capa 2). Sin red: sesion simulada."""

import os

import pandas as pd
import pytest
import requests

from data import apifootball


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _FakeSession:
    """Devuelve respuestas en cola, contando llamadas y guardando la ultima."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_url = None
        self.last_headers = None

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls += 1
        self.last_url = url
        self.last_headers = headers
        return self.responses.pop(0)


# --- host y cabecera de API-Sports -----------------------------------------

def test_get_usa_host_y_cabecera_de_apisports(tmp_path, monkeypatch):
    monkeypatch.setattr(apifootball, "CACHE_DIR", tmp_path)
    session = _FakeSession([_FakeResponse(200, {"response": []})])
    apifootball.get("status", api_key="k", session=session)
    assert session.last_url.startswith("https://v3.football.api-sports.io/")
    assert session.last_headers.get("x-apisports-key") == "k"
    assert "x-rapidapi-key" not in session.last_headers


# --- cache y reintentos ----------------------------------------------------

def test_get_cachea_y_no_repite_la_red(tmp_path, monkeypatch):
    monkeypatch.setattr(apifootball, "CACHE_DIR", tmp_path)
    session = _FakeSession([_FakeResponse(200, {"response": [1, 2]})])

    data = apifootball.get("fixtures", {"x": 1}, api_key="k", session=session)
    assert data["response"] == [1, 2]
    assert session.calls == 1
    assert apifootball.is_cached("fixtures", {"x": 1})  # ya quedo en cache

    again = apifootball.get("fixtures", {"x": 1}, api_key="k", session=session)
    assert again == data
    assert session.calls == 1  # no volvio a la red


def test_get_reintenta_en_429(tmp_path, monkeypatch):
    monkeypatch.setattr(apifootball, "CACHE_DIR", tmp_path)
    session = _FakeSession([_FakeResponse(429), _FakeResponse(200, {"response": []})])
    data = apifootball.get("fixtures", {"y": 1}, api_key="k", session=session, backoff_base=0)
    assert data == {"response": []}
    assert session.calls == 2


def test_get_falla_tras_agotar_reintentos(tmp_path, monkeypatch):
    monkeypatch.setattr(apifootball, "CACHE_DIR", tmp_path)
    session = _FakeSession([_FakeResponse(429) for _ in range(4)])
    with pytest.raises(requests.HTTPError):
        apifootball.get("fixtures", {"z": 1}, api_key="k", session=session, backoff_base=0, max_retries=4)
    assert session.calls == 4


def test_get_sin_clave_y_sin_cache_falla(tmp_path, monkeypatch):
    monkeypatch.setattr(apifootball, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("APISPORTS_KEY", raising=False)
    with pytest.raises(RuntimeError):
        apifootball.get("fixtures", {"q": 1})


def test_load_env_no_pisa_lo_existente(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("APISPORTS_KEY=fromfile\n# comentario\nOTRA=x\n", encoding="utf-8")
    monkeypatch.delenv("APISPORTS_KEY", raising=False)
    monkeypatch.setenv("OTRA", "ya")
    apifootball.load_env(env_file)
    assert os.environ["APISPORTS_KEY"] == "fromfile"
    assert os.environ["OTRA"] == "ya"  # no pisa una variable ya definida


# --- parsing (puro) --------------------------------------------------------

def test_parse_fixture_statistics():
    payload = {"response": [
        {"team": {"name": "Spain"}, "statistics": [
            {"type": "Corner Kicks", "value": 7}, {"type": "Total Shots", "value": 15},
            {"type": "Shots on Goal", "value": 6}, {"type": "Fouls", "value": 10},
            {"type": "Yellow Cards", "value": 2}, {"type": "Red Cards", "value": None},
            {"type": "Ball Possession", "value": "60%"},
        ]},
        {"team": {"name": "Italy"}, "statistics": [
            {"type": "Corner Kicks", "value": 3}, {"type": "Yellow Cards", "value": 3},
            {"type": "Red Cards", "value": 1},
        ]},
    ]}
    df = apifootball.parse_fixture_statistics(payload)
    assert set(df["team"]) == {"Spain", "Italy"}
    assert df[df["team"] == "Spain"]["corners"].iloc[0] == 7
    assert pd.isna(df[df["team"] == "Spain"]["red_cards"].iloc[0])  # None -> NaN
    assert df[df["team"] == "Italy"]["red_cards"].iloc[0] == 1
    assert "Ball Possession" not in df.columns


def test_parse_fixtures_trae_el_arbitro():
    payload = {"response": [
        {"fixture": {"id": 101, "date": "2026-06-20T18:00:00+00:00",
                     "referee": "Daniele Orsato", "status": {"short": "NS"}},
         "teams": {"home": {"name": "Brazil"}, "away": {"name": "Serbia"}}},
    ]}
    df = apifootball.parse_fixtures(payload)
    assert df.iloc[0]["fixture_id"] == 101
    assert df.iloc[0]["referee"] == "Daniele Orsato"
    assert df.iloc[0]["home_team"] == "Brazil"


def test_referee_card_averages():
    fixtures = pd.DataFrame([
        {"referee": "A", "total_cards": 4},
        {"referee": "A", "total_cards": 6},
        {"referee": "B", "total_cards": 2},
        {"referee": None, "total_cards": 5},  # sin arbitro: se excluye
    ])
    out = apifootball.referee_card_averages(fixtures)
    row_a = out[out["referee"] == "A"].iloc[0]
    assert row_a["cards_per_match"] == pytest.approx(5.0)
    assert row_a["matches"] == 2
    assert set(out["referee"]) == {"A", "B"}
