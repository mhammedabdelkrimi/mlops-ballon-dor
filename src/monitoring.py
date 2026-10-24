from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

BASE_DIR = Path(__file__).resolve().parent.parent
P = yaml.safe_load(open(BASE_DIR / "configs" / "params.yaml", encoding="utf-8"))

COLS = [
    "matchs_club_couverts", "minutes_club_couvertes", "buts_club_couverts",
    "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club",
    "matchs_ucl", "minutes_ucl", "buts_ucl", "passes_ucl",
    "club_saison_champion_ucl", "club_saison_champion_top5",
    "selection_championne_cdm", "selection_championne_euro_ou_copa",
    "meilleur_joueur_ucl", "meilleur_joueur_selection_majeure",
]

reference = pd.read_csv(BASE_DIR / P["data"]["train_path"])[COLS].astype(float)

logs = BASE_DIR / "logs" / "predictions.csv"
if logs.exists() and len(pd.read_csv(logs)) >= 5:
    current = pd.read_csv(logs)[COLS].astype(float)
    source = "requêtes réelles reçues par l'API"
else:
    current = pd.read_csv(BASE_DIR / P["data"]["valid_path"])[COLS].astype(float)
    source = "simulation (candidats 2026)"

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

(BASE_DIR / "reports").mkdir(exist_ok=True)
sortie = BASE_DIR / "reports" / f"drift_{date.today()}.html"
report.save_html(str(sortie))
print(f"Rapport généré ({source}) : {sortie}")