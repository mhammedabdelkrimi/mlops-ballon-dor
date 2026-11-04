# MLOps Ballon d'Or

Pipeline MLOps de prédiction du Ballon d'Or : DVC, MLflow, FastAPI, Docker,
GitHub Actions, Evidently. Le modèle (régression logistique) est volontairement
simple : l'objectif du projet est l'infrastructure.

## Architecture

    Git push (code ou données)
      → GitHub Actions : tests → entraînement (dvc repro) → safe deployment
      → image Docker publiée (GHCR) si le F1 est strictement supérieur
    DVC + stockage S3 : local (SeaweedFS) et DagsHub (pour la CI)
    MLflow : suivi des expériences · FastAPI : service et dashboard · Evidently : dérive

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

- Dashboard : http://localhost:8000/dashboard
- API : http://localhost:8000/docs (`/health`, `/predict`)
- Stockage S3 local : port 9000 (le bucket est créé automatiquement)

Si `/health` indique `model_loaded: false`, lance d'abord `dvc pull -r dagshub`
(ou `dvc repro`), puis relance la commande.

Après une modification du `Dockerfile` ou du code, reconstruis l'image :

    docker compose down
    docker compose up --build

## Dashboard

La page http://localhost:8000/dashboard a trois onglets :

- **Données & performances** : nombre de lignes, rang du vrai gagnant pour chaque
  édition, exploration des variables, et métriques de validation.
- **Classement des candidats 2026** : la probabilité de gagner de chaque joueur
  (sortie directe du modèle) et la chance relative entre candidats (part du total,
  qui fait 100 %).
- **Simulateur** : formulaire qui appelle `/predict`.

Le dashboard ne recalcule rien. Il lit `metrics/backtest.json` et
`metrics/classement_2026.json`, écrits par `src/train.py`. Ses chiffres sont donc
exactement ceux de l'entraînement.

Les métriques viennent d'une validation par édition (Leave-One-Edition-Out) : on
cache une édition, on entraîne sur les autres, et on regarde le rang donné au vrai
gagnant. Avec un seul gagnant par édition, Precision, Recall, F1 et Top-1 sont
égaux. Si une page affiche « introuvable », lance `dvc repro`.

## Commandes utiles

    dvc repro                                       # réentraîner
    pytest -v                                       # tests de l'API et du dashboard
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Pour voir les graphiques de l'exploration sur ton PC (par défaut, aucune fenêtre) :

    $env:SHOW_PLOTS="1"; python src/train.py

Rapport de dérive (environnement séparé, car Evidently n'est pas compatible pandas 3) :

    python -m venv venv-monitoring
    venv-monitoring\Scripts\Activate.ps1
    pip install -r requirements-monitoring.txt
    python src/monitoring.py

Le rapport HTML est écrit dans `reports/`. Un workflow GitHub le génère chaque lundi.

## Réglages (configs/params.yaml)

- `model.grid_search` : `true` = `train.py` cherche le meilleur `C` par validation
  par édition ; `false` = utilise `model.C` tel quel.
- `model.jeu_features`, `model.class_weight` : configuration retenue.

## Structure

    configs/params.yaml        hyperparamètres centralisés
    src/train.py               entraînement, GridSearchCV, suivi MLflow
    src/api/main.py            API FastAPI (/health, /predict)
    src/api/dashboard.py       routes du dashboard (lit les JSON de train.py)
    src/api/static/            page du dashboard (dashboard.html)
    src/compare.py             safe deployment
    src/monitoring.py          rapport Evidently (Data Drift)
    tests/test_api.py          tests pytest de l'API (erreur 400)
    tests/test_dashboard.py    tests pytest du dashboard
    metrics/                   metrics.json, backtest.json, classement_2026.json,
                               production.json
    images/                    logo du dashboard
    dvc.yaml                   pipeline DVC
    Dockerfile                 image de l'API (modèle, métriques, images, données)
    .github/workflows/         CI/CD et rapport hebdomadaire

## Règles du projet

- Aucun modèle ni CSV dans Git : tout passe par DVC. Seuls les petits fichiers JSON
  de `metrics/` sont dans Git.
- Déploiement uniquement si le F1 est strictement supérieur à celui en production.
- L'API renvoie 400 si les données sont mal formatées, 503 si le modèle est absent.
- Le pipeline se déclenche sur un push de code (src, configs...) ou de nouvelles données (.dvc).

## Secret GitHub requis

`DAGSHUB_TOKEN`