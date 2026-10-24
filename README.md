# MLOps Ballon d'Or

Pipeline MLOps de prédiction du Ballon d'Or : DVC, MLflow, FastAPI, Docker,
GitHub Actions, Evidently. Le modèle (régression logistique) est volontairement
simple : l'objectif du projet est l'infrastructure.

## Architecture

    Git push (code ou données)
      → GitHub Actions : tests → entraînement (dvc repro) → safe deployment
      → image Docker publiée (GHCR) si le F1 est strictement supérieur
    DVC + stockage S3 : local (SeaweedFS) et DagsHub (pour la CI)
    MLflow : suivi des expériences · FastAPI : service · Evidently : dérive

## Prérequis

Python 3.11, Git, Docker Desktop.

    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Récupérer les données et le modèle (DVC)

Les identifiants ne sont jamais dans Git. Configure-les une fois :

    dvc remote modify --local dagshub access_key_id TON_TOKEN
    dvc remote modify --local dagshub secret_access_key TON_TOKEN
    dvc pull -r dagshub

## Lancer l'architecture en une commande

    docker compose up --build

- API : http://localhost:8000/docs (`/health`, `/predict`)
- Stockage S3 local : port 9000 (le bucket est créé automatiquement)

Si `/health` indique `model_loaded: false`, lance d'abord `dvc pull -r dagshub`
(ou `dvc repro`), puis relance la commande.

## Commandes utiles

    dvc repro                                       # réentraîner
    pytest -v                                       # tests de l'API
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Rapport de dérive (environnement séparé, car Evidently n'est pas compatible pandas 3) :

    python -m venv venv-monitoring
    venv-monitoring\Scripts\Activate.ps1
    pip install -r requirements-monitoring.txt
    python src/monitoring.py

Le rapport HTML est écrit dans `reports/`. Un workflow GitHub le génère chaque lundi.

## Structure

    configs/params.yaml   hyperparamètres centralisés
    src/train.py          entraînement + suivi MLflow
    src/api/main.py       API FastAPI (/health, /predict)
    src/compare.py        safe deployment
    src/monitoring.py     rapport Evidently (Data Drift)
    tests/test_api.py     tests pytest (erreur 400)
    dvc.yaml              pipeline DVC
    .github/workflows/    CI/CD et rapport hebdomadaire

## Règles du projet

- Aucun modèle ni CSV dans Git : tout passe par DVC.
- Déploiement uniquement si le F1 est strictement supérieur à celui en production.
- L'API renvoie 400 si les données sont mal formatées, 503 si le modèle est absent.
- Le pipeline se déclenche sur un push de code (src, configs...) ou de nouvelles données (.dvc).

## Secret GitHub requis

`DAGSHUB_TOKEN`
