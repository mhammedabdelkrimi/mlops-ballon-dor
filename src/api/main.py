import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
LOG_PATH = BASE_DIR / "logs" / "predictions.csv"

app = FastAPI(title="API Ballon d'Or")
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class Joueur(BaseModel):
    poste: Literal["Attaquant", "Milieu", "Defenseur", "Gardien"]
    matchs_club_couverts: int = Field(..., ge=0)
    minutes_club_couvertes: int = Field(..., ge=0)
    buts_club_couverts: int = Field(..., ge=0)
    passes_club_couvertes: int = Field(..., ge=0)
    buts_par_90_club: float = Field(..., ge=0)
    passes_par_90_club: float = Field(..., ge=0)
    matchs_ucl: int = Field(..., ge=0)
    minutes_ucl: int = Field(..., ge=0)
    buts_ucl: int = Field(..., ge=0)
    passes_ucl: int = Field(..., ge=0)
    club_saison_champion_ucl: Literal[0, 1]
    club_saison_champion_top5: Literal[0, 1]
    selection_championne_cdm: Literal[0, 1]
    selection_championne_euro_ou_copa: Literal[0, 1]
    meilleur_joueur_ucl: Literal[0, 1]
    meilleur_joueur_selection_majeure: Literal[0, 1]


# Le PDF demande une erreur 400 (FastAPI renvoie 422 par défaut)
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    erreurs = [{"champ": ".".join(str(x) for x in e["loc"]), "message": e["msg"]}
               for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": erreurs})


def preparer(j: Joueur) -> pd.DataFrame:
    """Refait le même prétraitement que train.py."""
    d = j.model_dump()
    for col, (low, up) in bundle["iqr_limits"].items():          # clipping IQR
        d[col] = min(max(d[col], low), up)
    d["buts_ucl_log1p"] = math.log1p(d["buts_ucl"])               # log1p
    d["passes_ucl_log1p"] = math.log1p(d["passes_ucl"])
    d["matchs_club_couverts_yeo"] = float(                        # Yeo-Johnson
        bundle["yeo"].transform(
            pd.DataFrame({"matchs_club_couverts": [d["matchs_club_couverts"]]}))[0][0])
    ohe = bundle["ohe"]                                           # One-Hot du poste
    codes = ohe.transform(pd.DataFrame({"poste": [d["poste"]]}))[0]
    d.update(dict(zip(ohe.get_feature_names_out(["poste"]), codes)))
    manquantes = [f for f in bundle["features"] if f not in d]
    if manquantes:
        raise ValueError(f"Features non gérées par l'API : {manquantes}")
    return pd.DataFrame([d])[bundle["features"]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}


@app.post("/predict")
def predict(joueur: Joueur):
    if bundle is None:
        return JSONResponse(status_code=503, content={"detail": "Modèle non chargé"})
    try:
        X = preparer(joueur)
    except ValueError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    proba = float(bundle["model"].predict_proba(X)[0][1])

    # Journal des requêtes, utilisé par le monitoring de dérive
    LOG_PATH.parent.mkdir(exist_ok=True)
    nouveau = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        champs = list(joueur.model_dump().keys())
        if nouveau:
            w.writerow(["timestamp", *champs, "proba"])
        w.writerow([datetime.now().isoformat(), *joueur.model_dump().values(), proba])

    return {"probabilite_ballon_dor": proba, "prediction": int(proba >= 0.5)}