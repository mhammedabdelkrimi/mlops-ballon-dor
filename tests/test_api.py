from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

VALIDE = {
    "poste": "Attaquant", "matchs_club_couverts": 50, "minutes_club_couvertes": 4000,
    "buts_club_couverts": 30, "passes_club_couvertes": 10, "buts_par_90_club": 0.7,
    "passes_par_90_club": 0.2, "matchs_ucl": 12, "minutes_ucl": 1000, "buts_ucl": 8,
    "passes_ucl": 3, "club_saison_champion_ucl": 1, "club_saison_champion_top5": 1,
    "selection_championne_cdm": 0, "selection_championne_euro_ou_copa": 0,
    "meilleur_joueur_ucl": 0, "meilleur_joueur_selection_majeure": 0,
}


def test_health():
    assert client.get("/health").status_code == 200


def test_predict_valide():
    # 200 si le modèle est présent, 503 sinon (cas de la CI avant dvc pull)
    assert client.post("/predict", json=VALIDE).status_code in (200, 503)


def test_champs_manquants_renvoie_400():
    assert client.post("/predict", json={}).status_code == 400


def test_type_invalide_renvoie_400():
    assert client.post("/predict", json={**VALIDE, "buts_ucl": "abc"}).status_code == 400


def test_valeur_negative_renvoie_400():
    assert client.post("/predict", json={**VALIDE, "buts_ucl": -5}).status_code == 400


def test_poste_inconnu_renvoie_400():
    assert client.post("/predict", json={**VALIDE, "poste": "Arbitre"}).status_code == 400