import json

from fastapi.testclient import TestClient

import src.api.dashboard as dash
from src.api.main import app

client = TestClient(app)


def test_page_dashboard():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "Ballon d'Or Analytics" in r.text


def test_logo():
    assert client.get("/images/ballon-dor.png").status_code == 200


def test_overview():
    # 200 si metrics/backtest.json existe, 404 sinon (CI avant dvc repro)
    assert client.get("/dashboard/api/overview").status_code in (200, 404)


def test_ranking():
    assert client.get("/dashboard/api/ranking?limit=5").status_code in (200, 404)


def test_ranking_limit_invalide_renvoie_400():
    assert client.get("/dashboard/api/ranking?limit=0").status_code == 400


def test_overview_affiche_les_chiffres_de_train(tmp_path, monkeypatch):
    bt = {"config": {"n_features": 3}, "summary": {"top1": 0.5}, "per_edition": []}
    f = tmp_path / "backtest.json"
    f.write_text(json.dumps(bt), encoding="utf-8")
    monkeypatch.setattr(dash, "BACKTEST_PATH", f)
    r = client.get("/dashboard/api/overview")
    assert r.status_code == 200
    assert r.json()["backtest"]["summary"]["top1"] == 0.5


def test_ranking_affiche_le_classement_de_train(tmp_path, monkeypatch):
    rows = [
        {"rang": 1, "joueur": "Joueur A", "chance_pct": 20.0, "proba_brute": 0.4},
        {"rang": 2, "joueur": "Joueur B", "chance_pct": 10.0, "proba_brute": 0.2},
    ]
    f = tmp_path / "classement.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(dash, "CLASSEMENT_PATH", f)
    r = client.get("/dashboard/api/ranking?limit=1")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 1
    assert r.json()["rows"][0]["name"] == "Joueur A"