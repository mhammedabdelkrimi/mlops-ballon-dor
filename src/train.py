# ============================================================
# >>> MLOPS (ajout 1) : mode "sans fenêtre" PAR DÉFAUT
# Sur un serveur (GitHub Actions, Docker) il n'y a pas d'écran :
# un plt.show() resterait bloqué. Donc par défaut, aucune fenêtre.
# Pour voir les graphiques sur ton PC :
#     $env:SHOW_PLOTS="1"; python src/train.py
# Ce bloc doit rester tout en haut, AVANT d'importer pyplot / seaborn.
# ============================================================
import os
import matplotlib
if os.environ.get("SHOW_PLOTS") != "1":
    matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score      # AJOUT : métriques Precision / Recall / F1
from sklearn.model_selection import LeaveOneGroupOut, train_test_split   # CORRECTION : LeaveOneGroupOut ajouté
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone                                           # CORRECTION : import manquant
from sklearn.impute import SimpleImputer                                 # CORRECTION : import manquant
from sklearn.linear_model import LogisticRegression                      # CORRECTION : import manquant
from matplotlib import pyplot as plt
from pathlib import Path

# >>> MLOPS (ajout 2) : nouveaux imports
import json        # pour écrire metrics.json
import joblib      # pour sauvegarder le modèle (.pkl)
import yaml        # pour lire params.yaml
import mlflow      # le "cahier de laboratoire"
import mlflow.sklearn

# ============================================================
# 1. CHARGEMENT
# ============================================================
# Chemin du dossier principal du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# >>> MLOPS (ajout 3) : lecture de configs/params.yaml
# P est un dictionnaire : P["model"]["C"] donne 0.3, etc.
# encoding="utf-8" est nécessaire sur Windows à cause du mot "Prétraité".
with open(BASE_DIR / "configs" / "params.yaml", encoding="utf-8") as f:
    P = yaml.safe_load(f)

# Lecture des données d'entraînement et de test
# >>> MLOPS : le chemin vient maintenant de params.yaml
data = pd.read_csv(BASE_DIR / P["data"]["train_path"])

# Lecture des candidats de 2026 pour la validation
data_validation = pd.read_csv(BASE_DIR / P["data"]["valid_path"])

print(f"Shape : {data.shape}")
print(f"\nColonnes :\n{data.columns.tolist()}")
print("=== Types de données ===")
print(data.dtypes)
print("\n=== description des donnees ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(data.describe(include='all').T)

# Distribution de la cible
print("Distribution gagnant_reel:")
print(data['gagnant_reel'].value_counts(normalize=True).round(3))

fig, axes = plt.subplots(1, 1, figsize=(6, 4))
sns.countplot(data=data, x='gagnant_reel')
axes.set_title('Distribution gagnant_reel')
plt.show()

print(data.groupby("edition")["gagnant_reel"].sum())
print("____________________________________")

import math

num_cols = [
    "edition",
    "matchs_club_couverts",
    "minutes_club_couvertes",
    "buts_club_couverts",
    "passes_club_couvertes",
    "buts_par_90_club",
    "passes_par_90_club",
    "matchs_ucl",
    "minutes_ucl",
    "buts_ucl",
    "passes_ucl",
    "club_saison_champion_ucl",
    "club_saison_champion_top5",
    "selection_championne_cdm",
    "selection_championne_euro_ou_copa",
    "meilleur_joueur_ucl",
    "meilleur_joueur_selection_majeure",
]

# Séparer les variables binaires et les autres variables
binary_cols = [
    col for col in num_cols
    if set(data[col].dropna().unique()).issubset({0, 1})
    and len(data[col].dropna().unique()) <= 2
]

continuous_cols = [
    col for col in num_cols if col not in binary_cols
]

sns.set_theme(style="whitegrid")
# ============================================================
# 2. Afficher l'histogramme
# ============================================================

def afficher_histogrammes(columns, ncols=3):
    for start in range(0, len(columns), ncols):
        batch = columns[start:start + ncols]
        nrows = math.ceil(len(batch) / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 5 * nrows),
            squeeze=False
        )
        axes = axes.ravel()

        for i, col in enumerate(batch):
            ax = axes[i]
            values = data[col].dropna()

            sns.histplot(
                values,
                kde=len(values.unique()) > 5,
                ax=ax
            )

            skew_value = values.skew()
            ax.set_title(
                f"{col}\nSkewness = {skew_value:.2f}",
                fontsize=11
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Nombre de joueurs")

        for j in range(len(batch), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3)
        plt.show()
        plt.close("all")   # >>> MLOPS : libère la mémoire (utile en mode sans fenêtre)

def afficher_variables_binaires(columns, ncols=3):
    for start in range(0, len(columns), ncols):
        batch = columns[start:start + ncols]
        nrows = math.ceil(len(batch) / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 5 * nrows),
            squeeze=False
        )
        axes = axes.ravel()

        for i, col in enumerate(batch):
            ax = axes[i]
            sns.countplot(
                x=data[col],
                ax=ax
            )
            ax.set_title(f"Distribution de {col}", fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel("Nombre de joueurs")

        for j in range(len(batch), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3)
        plt.show()
        plt.close("all")   # >>> MLOPS

# 1. Variables numériques continues
afficher_histogrammes(continuous_cols)

# 2. Variables binaires : graphiques en barres
afficher_variables_binaires(binary_cols)

# ============================================================
# 3. afficher box plots
# ============================================================
# Boxplots pour visualiser les outliers
# Colonnes à examiner pour détecter les valeurs extrêmes
box_cols = [
    "matchs_club_couverts",
    "minutes_club_couvertes",
    "buts_club_couverts",
    "passes_club_couvertes",
    "buts_par_90_club",
    "passes_par_90_club",
    "matchs_ucl",
    "minutes_ucl",
    "buts_ucl",
    "passes_ucl"
]

sns.set_theme(style="whitegrid")

# Afficher 3 boxplots par figure
ncols = 3

for start in range(0, len(box_cols), ncols):
    cols = box_cols[start:start + ncols]
    nrows = math.ceil(len(cols) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 5 * nrows),
        squeeze=False
    )

    axes = axes.ravel()

    for i, col in enumerate(cols):
        values = data[col].dropna()

        sns.boxplot(
            y=values,
            ax=axes[i],
            width=0.4
        )

        axes[i].set_title(f"Boxplot : {col}", fontsize=11)
        axes[i].set_ylabel("Valeur")

    # Supprimer les emplacements inutilisés
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3)
    #plt.show()
    plt.close("all")   # >>> MLOPS

# Cardinalité des variables catégorielles

cat_cols = ["poste", "couverture_stats_club"]

for col in cat_cols:
    print(f"\n{'=' * 50}")
    print(f"Colonne : {col}")
    print(f"Nombre de valeurs uniques : {data[col].nunique()}")

    print("\nValeurs uniques :")
    print(data[col].dropna().unique())

    print("\nNombre d'occurrences de chaque valeur :")
    print(data[col].value_counts(dropna=False))

# ============================================================
# 4. Afficher heatMap de correlation
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Liste des features numériques
num_cols = [
    "edition",
    "matchs_club_couverts",
    "minutes_club_couvertes",
    "buts_club_couverts",
    "passes_club_couvertes",
    "buts_par_90_club",
    "passes_par_90_club",
    "matchs_ucl",
    "minutes_ucl",
    "buts_ucl",
    "passes_ucl",
    "club_saison_champion_ucl",
    "club_saison_champion_top5",
    "selection_championne_cdm",
    "selection_championne_euro_ou_copa",
    "meilleur_joueur_ucl",
    "meilleur_joueur_selection_majeure",
]

# 2. Ajouter la cible
target = "gagnant_reel"

# 3. Sélectionner les colonnes nécessaires
cols = num_cols + [target]
df_corr = data[cols].copy()

# 4. Calculer les corrélations
corr = df_corr.corr(numeric_only=True)

# 5. Afficher uniquement la corrélation de chaque feature avec la cible
plt.figure(figsize=(8, 10))

sns.heatmap(
    corr[[target]].drop(index=target),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5
)

plt.title("Corrélation des features numériques avec gagnant_reel")
plt.tight_layout()
plt.show()
plt.close("all")   # >>> MLOPS


# ============================================================
# CORRECTION 1 : FEATURES RELATIVES À L'ÉDITION
# (placé AVANT la création de X / y et le split, pour que les nouvelles
#  colonnes arrivent dans X)
# Le Ballon d'Or est une compétition entre les candidats d'une même année :
# ce qui compte, c'est la position d'un joueur par rapport aux autres
# candidats de son édition. Pas d'utilisation de la cible -> pas de fuite.
# ============================================================
cols_relatives_src = [
    "buts_club_couverts",
    "passes_club_couvertes",
    "buts_par_90_club",
    "passes_par_90_club",
    "buts_ucl",
    "passes_ucl",
    "minutes_ucl",
    "minutes_club_couvertes",
]

def ajouter_features_relatives(df, groupes):
    df = df.copy()
    for c in cols_relatives_src:
        g = df.groupby(groupes)[c]
        df[c + "_rk"] = g.rank(pct=True)             # rang en percentile dans l'édition
        df[c + "_gap"] = df[c] - g.transform("max")  # écart au meilleur de l'édition
    contributions = df["buts_club_couverts"] + df["passes_club_couvertes"]
    df["ga_club_rk"] = contributions.groupby(groupes).rank(pct=True)
    df["titres_total"] = (
        df["club_saison_champion_ucl"]
        + df["club_saison_champion_top5"]
        + df["selection_championne_cdm"]
        + df["selection_championne_euro_ou_copa"]
    )
    return df

data = ajouter_features_relatives(data, data["edition"])
data_validation = ajouter_features_relatives(
    data_validation, pd.Series(2026, index=data_validation.index)
)

# ============================================================
# 5. Deviser donnees
# ============================================================

# Préparer les données de validation : candidats 2026

# Garder les noms pour afficher les résultats ensuite
noms_validation = data_validation["joueur"].copy()

# X_validation contient les caractéristiques des candidats
X_validation = data_validation.drop(
    columns=["joueur", "gagnant_reel"],
    errors="ignore"
)

# 1. Les caractéristiques utilisées pour la prédiction
X = data.drop(columns=["joueur", "gagnant_reel"])

# 2. La cible à prédire : 0 ou 1
y = data["gagnant_reel"]

# Vérifier que les colonnes correspondent à celles de X
X_validation = X_validation.reindex(columns=X.columns)
print("<---------------------->")

print("Forme de X_validation :", X_validation.shape)
print("Colonnes manquantes dans le fichier 2026 :")
print(X.columns.difference(data_validation.columns).tolist())
print("<---------------------->")

# 3. Séparation des données
# >>> MLOPS : test_size et random_state viennent de params.yaml
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=P["split"]["test_size"],
    random_state=P["split"]["random_state"],
    stratify=y
)

# 4. Vérifier les dimensions et la proportion de gagnants
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
print(f"Gagnants train : {y_train.sum()} / {len(y_train)}")
print(f"Gagnants test : {y_test.sum()} / {len(y_test)}")
print(f"Proportion train : {y_train.mean():.3f}")
print(f"Proportion test : {y_test.mean():.3f}")

# ============================================================
# 6. Gestion des valeurs extrémes par IQR
# ============================================================
def iqr_clip(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower, upper), lower, upper


cols_iqr = [
    "matchs_club_couverts",
    "minutes_club_couvertes",
    "buts_ucl"
]

iqr_limits = {}

for col in cols_iqr:
    # Calcul des limites avec le train seulement
    X_train[col], low, up = iqr_clip(X_train[col])
    iqr_limits[col] = (float(low), float(up))   # >>> MLOPS : float() = valeur simple, facile à sauvegarder

    # Appliquer les mêmes limites au test
    X_test[col] = X_test[col].clip(lower=low, upper=up)

    # Appliquer les mêmes limites aux candidats 2026
    X_validation[col] = X_validation[col].clip(
        lower=low,
        upper=up
    )

    print(f"{col} : clipping entre {low:.1f} et {up:.1f}")


# Sauvegarder la Baseline AVANT les transformations
X_train_base = X_train.copy()
X_test_base = X_test.copy()

y_train_base = y_train.copy()
y_test_base = y_test.copy()

# ============================================================
# 7. TRANSFORMATION YEO-JOHNSON POUR l'asymétrie
# ============================================================
import numpy as np

# Colonnes à transformer
cols_log = ["buts_ucl", "passes_ucl"]

for col in cols_log:
    # log1p(x) = log(1 + x)
    X_train[col + "_log1p"] = np.log1p(X_train[col])
    X_test[col + "_log1p"] = np.log1p(X_test[col])
    X_validation[col + "_log1p"] = np.log1p(X_validation[col])

    print(f"\n{col}")
    print("Asymétrie avant :", X_train[col].skew())
    print("Asymétrie après :", X_train[col + "_log1p"].skew())

X_train = X_train.drop(columns=["buts_ucl", "passes_ucl"])
X_test = X_test.drop(columns=["buts_ucl", "passes_ucl"])
X_validation = X_validation.drop(columns=["buts_ucl", "passes_ucl"])

from sklearn.preprocessing import PowerTransformer

col = "matchs_club_couverts"

# Vérifier que la colonne existe
if col not in X_train.columns:
    raise ValueError(f"La colonne {col} est absente de X_train")

# Créer et entraîner le transformateur uniquement sur le train
yeo = PowerTransformer(
    method="yeo-johnson",
    standardize=False
)

X_train[col + "_yeo"] = yeo.fit_transform(
    X_train[[col]]
).ravel()

# Appliquer les mêmes paramètres au test et à la validation
X_test[col + "_yeo"] = yeo.transform(
    X_test[[col]]
).ravel()

X_validation[col + "_yeo"] = yeo.transform(
    X_validation[[col]]
).ravel()

print("Transformation Yeo-Johnson terminée.")

print("Skew avant :", X_train[col].skew())
print("Skew après :", X_train[col + "_yeo"].skew())


# ============================================================
# 8. Encoder les donnees String
# ============================================================
from sklearn.preprocessing import OneHotEncoder

# Colonnes catégorielles à encoder
# CORRECTION 2 : on n'encode que "poste".
# couverture_stats_club décrit la source des statistiques (et donc l'époque),
# pas le joueur ; c'est aussi elle qui provoquait l'avertissement
# "Found unknown categories" sur les candidats 2026.
ohe_cols = [
    "poste",
]

# Apprendre l'encodage uniquement sur le train
ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
    drop="first"
)

ohe_train = pd.DataFrame(
    ohe.fit_transform(X_train[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=X_train.index
)

ohe_test = pd.DataFrame(
    ohe.transform(X_test[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=X_test.index
)

ohe_validation = pd.DataFrame(
    ohe.transform(X_validation[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=X_validation.index
)

# Supprimer les colonnes catégorielles originales
X_train = X_train.drop(columns=ohe_cols)
X_test = X_test.drop(columns=ohe_cols)
X_validation = X_validation.drop(columns=ohe_cols)

# Ajouter les colonnes One-Hot
X_train = pd.concat([X_train, ohe_train], axis=1)
X_test = pd.concat([X_test, ohe_test], axis=1)
X_validation = pd.concat([X_validation, ohe_validation], axis=1)

# CORRECTION : cette liste était utilisée plus bas sans avoir été définie
ohe_feature_cols = ohe.get_feature_names_out(ohe_cols).tolist()

print("One-Hot features créées :")
print(ohe_feature_cols)

print("<<----------------------------->>")

# ============================================================
# 9. REGROUPER LES DONNÉES (train + test = toutes les éditions étiquetées)
# ============================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])
groupes_all = data.loc[X_all.index, "edition"]

print(f"Données regroupées : {X_all.shape[0]} joueurs | {int(y_all.sum())} gagnants | "
      f"{groupes_all.nunique()} éditions")
print("Valeurs manquantes :", int(X_all.isnull().sum().sum()))

resume_editions = (
    pd.DataFrame({"edition": groupes_all, "gagnant": y_all})
    .groupby("edition")["gagnant"].agg(candidats="count", gagnants="sum")
)
print("\nTableau des éditions :")
print(resume_editions.to_string())

# ============================================================
# AJOUT : COLONNES BRUTES POUR LA VRAIE BASELINE
# `data` et `data_validation` n'ont jamais été clippés ni transformés :
# ils contiennent encore les valeurs d'origine du CSV.
# On copie ces valeurs dans X_all / X_validation avec le suffixe "_brut".
# ============================================================
cols_brutes = [
    "matchs_club_couverts", "minutes_club_couvertes", "buts_club_couverts",
    "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club",
    "matchs_ucl", "minutes_ucl", "buts_ucl", "passes_ucl",
]
for c in cols_brutes:
    X_all[c + "_brut"] = data.loc[X_all.index, c]
    X_validation[c + "_brut"] = data_validation.loc[X_validation.index, c]

brut_cols = [c + "_brut" for c in cols_brutes]

# ============================================================
# 10. FEATURES : on liste les colonnes explicitement
# (on exclut 'edition' et 'couverture_stats_club' : ce ne sont pas des features)
# ============================================================
raw_num_cols = [
    "minutes_club_couvertes", "buts_club_couverts", "passes_club_couvertes",
    "buts_par_90_club", "passes_par_90_club", "matchs_ucl", "minutes_ucl",
]
transformed_cols = ["matchs_club_couverts_yeo", "buts_ucl_log1p", "passes_ucl_log1p"]
binary_feature_cols = [
    "club_saison_champion_ucl", "club_saison_champion_top5",
    "selection_championne_cdm", "selection_championne_euro_ou_copa",
    "meilleur_joueur_ucl", "meilleur_joueur_selection_majeure",
]
relative_cols = (
    [c + "_rk" for c in cols_relatives_src]
    + [c + "_gap" for c in cols_relatives_src]
    + ["ga_club_rk", "titres_total"]
)

# CORRECTION 2 : seulement les colonnes One-Hot de "poste"
ohe_poste_cols = [c for c in ohe_feature_cols if c.startswith("poste_")]

# 3 configurations comparées :
#  - Baseline brute : valeurs d'origine du CSV + encodage du poste (aucun clipping, log, Yeo-Johnson)
#  - Prétraité      : valeurs extrêmes (IQR) + asymétrie (log1p, Yeo-Johnson) + encodage
#  - Prétraité + FE : Prétraité + variables relatives à l'édition (feature engineering)
jeux_features = {
    "Baseline brute": brut_cols + binary_feature_cols + ohe_poste_cols,
    "Prétraité": raw_num_cols + transformed_cols + binary_feature_cols + ohe_poste_cols,
    "Prétraité + FE": (raw_num_cols + transformed_cols + relative_cols
                       + binary_feature_cols + ohe_poste_cols),
}
# brut_cols est ajouté pour que la baseline brute passe aussi par Imputer + RobustScaler
# (le scaler fait partie du modèle : sans lui la régression logistique converge mal)
cols_continues = set(raw_num_cols + transformed_cols + relative_cols + brut_cols)

for nom, cols in jeux_features.items():
    manquantes = [c for c in cols if c not in X_all.columns]
    if manquantes:
        raise ValueError(f"{nom} : colonnes absentes {manquantes}")
    print(f"{nom:<22}: {len(cols)} features")

# ============================================================
# 11. SCALING (RobustScaler) sur les données regroupées + tableau
# Les colonnes 0/1 (binaires, One-Hot) ne sont pas scalées.
# Ce tableau sert à VOIR l'effet du scaling (médiane -> 0, IQR -> 1).
# ============================================================
cols_a_scaler = [c for c in jeux_features["Prétraité + FE"] if c in cols_continues]
scaler = RobustScaler()
X_all_scaled_demo = pd.DataFrame(
    scaler.fit_transform(X_all[cols_a_scaler]),
    columns=cols_a_scaler, index=X_all.index,
)

def iqr(df):
    return df.quantile(0.75) - df.quantile(0.25)

tableau_scaling = pd.DataFrame({
    "mediane_avant": X_all[cols_a_scaler].median(),
    "IQR_avant": iqr(X_all[cols_a_scaler]),
    "mediane_apres": X_all_scaled_demo.median(),
    "IQR_apres": iqr(X_all_scaled_demo),
})
print("\n===== TABLEAU DU SCALING (RobustScaler, données regroupées) =====")
print(tableau_scaling.round(3).to_string())
print("✅ Scaling terminé (RobustScaler)")

# ============================================================
# 12. MODÈLE : régression logistique dans un Pipeline
# Le RobustScaler est refait DANS le Pipeline, à chaque pli, sur les seules
# éditions d'entraînement : l'édition cachée ne "voit" pas le scaler.
# Le modèle final (étape 14) est entraîné sur toutes les données regroupées.
# ============================================================
# >>> MLOPS : la valeur par défaut de C vient de params.yaml
def construire_modele(colonnes, class_weight=None, C=P["model"]["C"]):
    continues = [c for c in colonnes if c in cols_continues]
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", RobustScaler())]), continues)],
        remainder="passthrough",
    )
    return Pipeline([
        ("prep", prep),
        ("lr", LogisticRegression(C=C, class_weight=class_weight,
                                  max_iter=3000, random_state=42)),
    ])

# ============================================================
# 13. ÉVALUATION PAR ÉDITION : BASELINE BRUTE vs PRÉTRAITÉ vs PRÉTRAITÉ + FE
# ============================================================
def evaluer_par_edition(colonnes, class_weight):
    Xc = X_all[colonnes]
    modele = construire_modele(colonnes, class_weight)
    proba = pd.Series(index=Xc.index, dtype=float)
    for tr, va in LeaveOneGroupOut().split(Xc, y_all, groupes_all):
        m = clone(modele).fit(Xc.iloc[tr], y_all.iloc[tr])
        proba.iloc[va] = m.predict_proba(Xc.iloc[va])[:, 1]

    part = proba / proba.groupby(groupes_all).transform("sum")
    lignes = []
    for ed, idx in groupes_all.groupby(groupes_all).groups.items():
        gagnant = idx[y_all.loc[idx].values == 1]
        if len(gagnant) == 0:
            continue
        g = gagnant[0]
        lignes.append({
            "edition": ed,
            "gagnant": data.loc[g, "joueur"],
            "rang": int((proba.loc[idx] >= proba.loc[g]).sum()),
            "part_gagnant": part.loc[g],
        })
    det = pd.DataFrame(lignes).set_index("edition")

    # AJOUT : Precision / Recall / F1
    # "Le modèle annonce un gagnant" = le joueur classé 1er de chaque édition.
    # Il y a donc 1 annonce et 1 vrai gagnant par édition : Precision = Recall = F1 = Top-1.
    y_pred = pd.Series(0, index=Xc.index)
    for ed, idx in groupes_all.groupby(groupes_all).groups.items():
        y_pred.loc[proba.loc[idx].idxmax()] = 1
    y_vrai = y_all.loc[Xc.index]

    resume = {
        "Top-1": (det["rang"] == 1).mean(),
        "Top-3": (det["rang"] <= 3).mean(),
        "Rang moyen": det["rang"].mean(),
        "Part moy. du gagnant (%)": 100 * det["part_gagnant"].mean(),
        "Precision": precision_score(y_vrai, y_pred, zero_division=0),
        "Recall": recall_score(y_vrai, y_pred, zero_division=0),
        "F1-score": f1_score(y_vrai, y_pred, zero_division=0),
    }
    return resume, det

resultats, details = {}, {}
for nom_jeu, cols in jeux_features.items():
    for cw in (None, "balanced"):
        nom = f"{nom_jeu} | class_weight={cw}"
        resultats[nom], details[nom] = evaluer_par_edition(cols, cw)

df_res = pd.DataFrame(resultats).T.sort_values(
    ["Rang moyen", "Part moy. du gagnant (%)"], ascending=[True, False])
print("\n===== BASELINE BRUTE vs PRÉTRAITÉ vs PRÉTRAITÉ + FE (validation par édition) =====")
print(df_res.round(3).to_string())

meilleur_auto = df_res.index[0]
print(f"\nPremière ligne du tableau (tri automatique) : {meilleur_auto}")

# Choix manuel : Prétraité et Baseline brute sont à égalité (Top-1, Top-3, rang moyen).
# On retient Prétraité pour sa robustesse aux valeurs extrêmes sur les nouvelles données.
# >>> MLOPS : ce choix vient maintenant de params.yaml (jeu_features + class_weight)
meilleur = f"{P['model']['jeu_features']} | class_weight={P['model']['class_weight']}"
print(f"Configuration retenue : {meilleur}")
print("\nRang du vrai gagnant par édition (1 = trouvé) :")
print(details[meilleur].round(3).to_string())

# ============================================================
# 14. MODÈLE FINAL ET CLASSEMENT 2026
# ============================================================
nom_jeu, cw_txt = meilleur.split(" | class_weight=")
cw = None if cw_txt == "None" else cw_txt
colonnes_finales = jeux_features[nom_jeu]

# AJOUT : afficher exactement ce que le modèle final utilise
print("\n===== MODÈLE FINAL UTILISÉ POUR 2026 =====")
print(f"Jeu de variables : {nom_jeu}")
print(f"class_weight     : {cw}")
print(f"Nombre de variables : {len(colonnes_finales)}")
print("Variables :", colonnes_finales)

modele_final = construire_modele(colonnes_finales, cw).fit(X_all[colonnes_finales], y_all)
proba_2026 = modele_final.predict_proba(X_validation[colonnes_finales])[:, 1]

classement = pd.DataFrame({"joueur": noms_validation.values, "proba_brute": proba_2026})
classement["chance_%"] = 100 * classement["proba_brute"] / classement["proba_brute"].sum()
classement = classement.sort_values("proba_brute", ascending=False).reset_index(drop=True)
classement["rang"] = classement.index + 1

print("\n===== TOP 10 CANDIDATS 2026 =====")
print(classement.head(10)[["rang", "joueur", "chance_%"]].round(1).to_string(index=False))
print("\nTOP 3 annoncé :", ", ".join(classement["joueur"].head(3)))

# ============================================================
# >>> MLOPS (ajout 4) : 15. SAUVEGARDE + SUIVI MLFLOW
# Tout ce qui suit est nouveau. Il transforme ton script en
# "entraînement traçable" : il laisse des fichiers et un historique.
# ============================================================

# --- 15.1 Les métriques de la configuration retenue -------------------
# Les noms sont simplifiés (top1, f1...) car MLflow refuse des caractères
# comme "%" ou "(" dans les noms de métriques.
r = resultats[meilleur]

metrics = {
    "top1": float(r["Top-1"]),
    "top3": float(r["Top-3"]),
    "rang_moyen": float(r["Rang moyen"]),
    "part_gagnant_pct": float(r["Part moy. du gagnant (%)"]),
    "precision": float(r["Precision"]),
    "recall": float(r["Recall"]),
    "f1": float(r["F1-score"]),
}

# --- 15.2 Le "bundle" : le modèle + tous les outils de prétraitement ----
# L'API devra refaire EXACTEMENT les mêmes transformations :
#   clipping IQR -> iqr_limits | log1p -> formule simple | Yeo-Johnson -> yeo
#   One-Hot du poste -> ohe    | features relatives -> cols_relatives_src
bundle = {
    "model": modele_final,
    "ohe": ohe,
    "yeo": yeo,
    "iqr_limits": iqr_limits,
    "features": colonnes_finales,
    "cols_relatives_src": cols_relatives_src,
}

# --- 15.3 Sauvegarde sur le disque -------------------------------------
os.makedirs(BASE_DIR / "models", exist_ok=True)
os.makedirs(BASE_DIR / "metrics", exist_ok=True)

joblib.dump(bundle, BASE_DIR / P["paths"]["model"])               # -> models/model.pkl
with open(BASE_DIR / P["paths"]["metrics"], "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)                               # -> metrics/metrics.json

# --- 15.4 MLflow : le cahier de laboratoire ------------------------------
mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment(P["mlflow"]["experiment"])

with mlflow.start_run():                            # 1 exécution = 1 run
    mlflow.log_params({
        "C": P["model"]["C"],
        "class_weight": str(cw),
        "jeu_features": nom_jeu,
        "nb_features": len(colonnes_finales),
        "test_size": P["split"]["test_size"],
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(modele_final, name="model",
                         serialization_format="cloudpickle")   # format classique, accepte tous les types            # le modèle au format MLflow
    mlflow.log_artifact(str(BASE_DIR / P["paths"]["model"]))        # le bundle complet
    mlflow.log_artifact(str(BASE_DIR / P["paths"]["metrics"]))      # metrics.json

print("\n✅ Modèle sauvegardé, métriques enregistrées dans MLflow")
print(metrics)