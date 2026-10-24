import pandas as pd
import requests

COLS = ["poste", "matchs_club_couverts", "minutes_club_couvertes", "buts_club_couverts",
        "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club", "matchs_ucl",
        "minutes_ucl", "buts_ucl", "passes_ucl", "club_saison_champion_ucl",
        "club_saison_champion_top5", "selection_championne_cdm",
        "selection_championne_euro_ou_copa", "meilleur_joueur_ucl",
        "meilleur_joueur_selection_majeure"]

df = pd.read_csv("data/raw/candidats_ballon_dor_2026.csv")[COLS]
for ligne in df.to_dict(orient="records"):
    r = requests.post("http://localhost:8000/predict", json=ligne)
    print(r.status_code, r.json())