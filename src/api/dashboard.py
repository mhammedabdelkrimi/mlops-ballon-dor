"""Routes du dashboard : page web + endpoints de données.

Règle : AUCUN chiffre n'est recalculé ici. Tout vient des fichiers écrits par
src/train.py (metrics/backtest.json et metrics/classement_2026.json), donc les
chiffres du dashboard sont exactement ceux du notebook / de train.py.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

# ───────────────────────── CONFIG ─────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "entrainement_ballon_dor_2014_2025.csv"
CANDIDATES_PATH = BASE_DIR / "data" / "raw" / "candidats_ballon_dor_2026.csv"
BACKTEST_PATH = BASE_DIR / "metrics" / "backtest.json"
CLASSEMENT_PATH = BASE_DIR / "metrics" / "classement_2026.json"
TARGET = "gagnant_reel"
SEASON_COL = "edition"
# ──────────────────────────────────────────────────────────

STATIC = Path(__file__).parent / "static"
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@lru_cache(maxsize=1)
def _bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@lru_cache(maxsize=8)
def _csv(path: str):
    p = Path(path)
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def _json(path: Path):
    """Relu à chaque appel : après un nouveau `dvc repro`, la page est à jour."""
    if not path.exists():
        raise HTTPException(
            404, f"{path.name} introuvable : lance `dvc repro` (ou `python src/train.py`).")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("", include_in_schema=False)
def page():
    return FileResponse(STATIC / "dashboard.html")


@router.get("/api/overview")
def overview():
    bt = _json(BACKTEST_PATH)
    df = _csv(str(DATA_PATH))
    cand = _csv(str(CANDIDATES_PATH))
    b = _bundle()

    labelled = {"rows": 0, "positives": 0, "negatives": 0, "period": "n/a"}
    numeric = []
    if df is not None:
        labelled = {
            "rows": int(len(df)),
            "positives": int((df[TARGET] == 1).sum()),
            "negatives": int((df[TARGET] == 0).sum()),
            "period": f"{df[SEASON_COL].min()}–{df[SEASON_COL].max()}",
        }
        numeric = [c for c in df.select_dtypes("number").columns
                   if c not in (TARGET, SEASON_COL)]

    cfg = dict(bt["config"])
    if b is not None:
        cfg["n_features"] = len(b["features"])

    return {
        "config": cfg,
        "datasets": {
            "labelled": labelled,
            "candidates": {"rows": 0 if cand is None else int(len(cand))},
        },
        "backtest": {"summary": bt["summary"], "per_edition": bt["per_edition"]},
        "numeric_columns": numeric,
    }


@router.get("/api/ranking")
def ranking(season: str | None = None, limit: int = Query(10, ge=1, le=100)):
    rows = _json(CLASSEMENT_PATH)[:limit]
    return {
        "seasons": ["2026"],
        "rows": [{
            "rank": r["rang"],
            "name": r["joueur"],
            "season": "2026",
            "proba": r["proba_brute"],          # probabilité réelle donnée par le modèle
            "part": r["chance_pct"],            # part parmi les candidats (somme = 100 %)
            "actual": None,                     # 2026 pas encore joué
        } for r in rows],
    }


@router.get("/api/distribution")
def distribution(feature: str, bins: int = Query(25, ge=5, le=60)):
    df = _csv(str(DATA_PATH))
    if df is None or feature not in df.columns:
        raise HTTPException(404, "Variable introuvable")
    col = df[feature].dropna()
    edges = np.histogram_bin_edges(col, bins=bins)
    result = {"edges": [round(float(e), 3) for e in edges], "series": {}}
    for cls, label in [(0, "Non lauréat"), (1, "Lauréat")]:
        sub = df.loc[df[TARGET] == cls, feature].dropna()
        result["series"][label] = np.histogram(sub, bins=edges)[0].tolist()
    return result