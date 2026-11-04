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

import math
import json        # pour écrire metrics.json, backtest.json, classement_2026.json
import joblib      # pour sauvegarder le modèle (.pkl)
import yaml        # pour lire params.yaml
import mlflow      # le "cahier de laboratoire"
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

# ============================================================
# 1. CHARGEMENT
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# P est un dictionnaire : P["model"]["C"] donne 0.3, etc.
# encoding="utf-8" est nécessaire sur Windows à cause du mot "Prétraité".
with open(BASE_DIR / "configs" / "params.yaml", encoding="utf-8") as f:
    P = yaml.safe_load(f)

data = pd.read_csv(BASE_DIR / P["data"]["train_path"])
data_validation = pd.read_csv(BASE_DIR / P["data"]["valid_path"])   # candidats 2026

print(f"Shape : {data.shape}")
print(f"\nColonnes :\n{data.columns.tolist()}")
print("=== Types de données ===")
print(data.dtypes)
print("\n=== description des donnees ===")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
print(data.describe(include="all").T)

print("Distribution gagnant_reel:")
print(data["gagnant_reel"].value_counts(normalize=True).round(3))

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=data, x="gagnant_reel", ax=ax)
ax.set_title("Distribution gagnant_reel")
plt.show()
plt.close("all")

print("\nNombre de gagnants par édition :")
print(data.groupby("edition")["gagnant_reel"].sum())
print("____________________________________")

num_cols = [
    "edition",
    "matchs_club_couverts", "minutes_club_couvertes", "buts_club_couverts",
    "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club",
    "matchs_ucl", "minutes_ucl", "buts_ucl", "passes_ucl",
    "club_saison_champion_ucl", "club_saison_champion_top5",
    "selection_championne_cdm", "selection_championne_euro_ou_copa",
    "meilleur_joueur_ucl", "meilleur_joueur_selection_majeure",
]

binary_cols = [
    col for col in num_cols
    if set(data[col].dropna().unique()).issubset({0, 1})
    and len(data[col].dropna().unique()) <= 2
]
continuous_cols = [col for col in num_cols if col not in binary_cols]
print("Binaires  :", binary_cols)
print("Continues :", continuous_cols)

sns.set_theme(style="whitegrid")

# ============================================================
# 2. EXPLORATION : skewness + histogrammes
# ============================================================
# AJOUT (notebook) : tableau des skewness avec interprétation
def interpreter_skew(s):
    if abs(s) < 0.5:
        return "Symétrique"
    elif abs(s) < 1:
        return "Asymétrie modérée"
    return "Asymétrie FORTE"

skew_table = pd.DataFrame({"skewness": data[continuous_cols].skew()})
skew_table["sens"] = np.where(skew_table["skewness"] > 0, "droite", "gauche")
skew_table["interprétation"] = skew_table["skewness"].apply(interpreter_skew)
skew_table = skew_table.sort_values("skewness", key=abs, ascending=False)
print("\n===== TABLEAU DES SKEWNESS =====")
print(skew_table.round(2).to_string())


def afficher_histogrammes(columns, ncols=3):
    for start in range(0, len(columns), ncols):
        batch = columns[start:start + ncols]
        nrows = math.ceil(len(batch) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        axes = axes.ravel()
        for i, col in enumerate(batch):
            values = data[col].dropna()
            sns.histplot(values, kde=len(values.unique()) > 5, ax=axes[i])
            axes[i].set_title(f"{col}\nSkewness = {values.skew():.2f}", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Nombre de joueurs")
        for j in range(len(batch), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(pad=3)
        plt.show()
        plt.close("all")


def afficher_variables_binaires(columns, ncols=3):
    for start in range(0, len(columns), ncols):
        batch = columns[start:start + ncols]
        nrows = math.ceil(len(batch) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        axes = axes.ravel()
        for i, col in enumerate(batch):
            sns.countplot(x=data[col], ax=axes[i])
            axes[i].set_title(f"Distribution de {col}", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Nombre de joueurs")
        for j in range(len(batch), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(pad=3)
        plt.show()
        plt.close("all")


afficher_histogrammes(continuous_cols)
afficher_variables_binaires(binary_cols)

# ============================================================
# 3. BOXPLOTS (détection des valeurs extrêmes)
# ============================================================
box_cols = [
    "matchs_club_couverts", "minutes_club_couvertes", "buts_club_couverts",
    "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club",
    "matchs_ucl", "minutes_ucl", "buts_ucl", "passes_ucl",
]
ncols = 3
for start in range(0, len(box_cols), ncols):
    cols = box_cols[start:start + ncols]
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), squeeze=False)
    axes = axes.ravel()
    for i, col in enumerate(cols):
        sns.boxplot(y=data[col].dropna(), ax=axes[i], width=0.4)
        axes[i].set_title(f"Boxplot : {col}", fontsize=11)
        axes[i].set_ylabel("Valeur")
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(pad=3)
    plt.close("all")

cat_cols = ["poste", "couverture_stats_club"]
for col in cat_cols:
    print(f"\n{'=' * 50}")
    print(f"Colonne : {col}")
    print(f"Nombre de valeurs uniques : {data[col].nunique()}")
    print("\nValeurs uniques :", data[col].dropna().unique())
    print("\nNombre d'occurrences :")
    print(data[col].value_counts(dropna=False))

# ============================================================
# 4. CORRÉLATION AVEC LA CIBLE
# ============================================================
target = "gagnant_reel"
corr = data[num_cols + [target]].corr(numeric_only=True)

plt.figure(figsize=(8, 10))
sns.heatmap(corr[[target]].drop(index=target), annot=True, cmap="coolwarm",
            fmt=".2f", vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title("Corrélation des features numériques avec gagnant_reel")
plt.tight_layout()
plt.show()
plt.close("all")

# ============================================================
# 5. FEATURES RELATIVES À L'ÉDITION (feature engineering)
# Le Ballon d'Or est une compétition entre les candidats d'une même année :
# on compare chaque joueur aux autres candidats de son édition.
# La cible n'est pas utilisée -> pas de fuite de données.
# ============================================================
cols_relatives_src = [
    "buts_club_couverts", "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club",
    "buts_ucl", "passes_ucl", "minutes_ucl", "minutes_club_couvertes",
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
        df["club_saison_champion_ucl"] + df["club_saison_champion_top5"]
        + df["selection_championne_cdm"] + df["selection_championne_euro_ou_copa"]
    )
    return df


data = ajouter_features_relatives(data, data["edition"])
data_validation = ajouter_features_relatives(
    data_validation, pd.Series(2026, index=data_validation.index)
)

# AJOUT (notebook) : récapitulatif des nouvelles colonnes
# (ga_club_rk finit déjà par "_rk" : elle est donc comptée dans cols_rk)
cols_rk = [c for c in data.columns if c.endswith("_rk")]
cols_gap = [c for c in data.columns if c.endswith("_gap")]
nouvelles_cols = cols_rk + cols_gap + ["titres_total"]
print(f"\nFeatures relatives : {len(cols_rk)} _rk, {len(cols_gap)} _gap, "
      f"+ titres_total = {len(nouvelles_cols)} nouvelles colonnes")

# ============================================================
# 6. SÉPARATION X / y ET TRAIN / TEST
# ============================================================
noms_validation = data_validation["joueur"].copy()
X_validation = data_validation.drop(columns=["joueur", "gagnant_reel"], errors="ignore")

X = data.drop(columns=["joueur", "gagnant_reel"])
y = data["gagnant_reel"]

X_validation = X_validation.reindex(columns=X.columns)
print("<---------------------->")
print("Forme de X_validation :", X_validation.shape)
print("Colonnes manquantes dans le fichier 2026 :",
      X.columns.difference(data_validation.columns).tolist())
print("<---------------------->")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=P["split"]["test_size"],
    random_state=P["split"]["random_state"],
    stratify=y,
)

print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
print(f"Gagnants train : {y_train.sum()} / {len(y_train)}")
print(f"Gagnants test : {y_test.sum()} / {len(y_test)}")
print(f"Proportion train : {y_train.mean():.3f} | test : {y_test.mean():.3f}")

# ============================================================
# 7. VALEURS EXTRÊMES : clipping IQR (limites calculées sur le train seulement)
# ============================================================
def iqr_clip(series, factor=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    return series.clip(lower, upper), lower, upper


cols_iqr = ["matchs_club_couverts", "minutes_club_couvertes", "buts_ucl"]
iqr_limits = {}

for col in cols_iqr:
    X_train[col], low, up = iqr_clip(X_train[col])
    iqr_limits[col] = (float(low), float(up))     # float() = valeur simple, facile à sauvegarder
    X_test[col] = X_test[col].clip(lower=low, upper=up)
    X_validation[col] = X_validation[col].clip(lower=low, upper=up)
    print(f"{col} : clipping entre {low:.1f} et {up:.1f}")

# Sauvegarde AVANT les transformations
X_train_base, X_test_base = X_train.copy(), X_test.copy()
y_train_base, y_test_base = y_train.copy(), y_test.copy()

# ============================================================
# 8. ASYMÉTRIE : log1p puis Yeo-Johnson
# ============================================================
cols_log = ["buts_ucl", "passes_ucl"]

for col in cols_log:
    X_train[col + "_log1p"] = np.log1p(X_train[col])
    X_test[col + "_log1p"] = np.log1p(X_test[col])
    X_validation[col + "_log1p"] = np.log1p(X_validation[col])
    print(f"{col} -> skew avant : {X_train[col].skew():.3f} | "
          f"après : {X_train[col + '_log1p'].skew():.3f}")

X_train = X_train.drop(columns=cols_log)
X_test = X_test.drop(columns=cols_log)
X_validation = X_validation.drop(columns=cols_log)

col = "matchs_club_couverts"
if col not in X_train.columns:
    raise ValueError(f"La colonne {col} est absente de X_train")

yeo = PowerTransformer(method="yeo-johnson", standardize=False)
X_train[col + "_yeo"] = yeo.fit_transform(X_train[[col]]).ravel()   # fit sur le train seulement
X_test[col + "_yeo"] = yeo.transform(X_test[[col]]).ravel()
X_validation[col + "_yeo"] = yeo.transform(X_validation[[col]]).ravel()

print("Yeo-Johnson terminé.")
print("Skew avant :", round(X_train[col].skew(), 3))
print("Skew après :", round(X_train[col + "_yeo"].skew(), 3))

# ============================================================
# 9. ENCODAGE DU POSTE (One-Hot)
# On n'encode que "poste" : couverture_stats_club décrit la source des stats.
# ============================================================
ohe_cols = ["poste"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
noms_ohe = ohe.fit(X_train[ohe_cols]).get_feature_names_out(ohe_cols)


def encoder(Xdf):
    enc = pd.DataFrame(ohe.transform(Xdf[ohe_cols]), columns=noms_ohe, index=Xdf.index)
    return pd.concat([Xdf.drop(columns=ohe_cols), enc], axis=1)


X_train = encoder(X_train)
X_test = encoder(X_test)
X_validation = encoder(X_validation)

ohe_feature_cols = list(noms_ohe)
print("One-Hot features créées :", ohe_feature_cols)
print("<<----------------------------->>")

# ============================================================
# 10. REGROUPEMENT (train + test = toutes les éditions étiquetées)
# ============================================================
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

# Colonnes BRUTES (valeurs d'origine du CSV) pour la vraie baseline
cols_brutes = [
    "matchs_club_couverts", "minutes_club_couvertes", "buts_club_couverts",
    "passes_club_couvertes", "buts_par_90_club", "passes_par_90_club",
    "matchs_ucl", "minutes_ucl", "buts_ucl", "passes_ucl",
]
for c in cols_brutes:
    X_all[c + "_brut"] = data.loc[X_all.index, c]
    X_validation[c + "_brut"] = data_validation.loc[X_validation.index, c]

brut_cols = [c + "_brut" for c in cols_brutes]

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
ohe_poste_cols = [c for c in ohe_feature_cols if c.startswith("poste_")]

# 3 configurations comparées :
#  - Baseline brute : valeurs d'origine du CSV (aucun clipping, log, Yeo-Johnson)
#  - Prétraité      : IQR + log1p + Yeo-Johnson + encodage
#  - Prétraité + FE : Prétraité + variables relatives à l'édition
jeux_features = {
    "Baseline brute": brut_cols + binary_feature_cols + ohe_poste_cols,
    "Prétraité": raw_num_cols + transformed_cols + binary_feature_cols + ohe_poste_cols,
    "Prétraité + FE": (raw_num_cols + transformed_cols + relative_cols
                       + binary_feature_cols + ohe_poste_cols),
}
cols_continues = set(raw_num_cols + transformed_cols + relative_cols + brut_cols)

for nom, cols in jeux_features.items():
    manquantes = [c for c in cols if c not in X_all.columns]
    if manquantes:
        raise ValueError(f"{nom} : colonnes absentes {manquantes}")
    print(f"{nom:<22}: {len(cols)} features")

# ============================================================
# 11. SCALING (RobustScaler) : démonstration
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
print("\n===== TABLEAU DU SCALING (RobustScaler) =====")
print(tableau_scaling.round(3).to_string())

# ============================================================
# 12. MODÈLE : régression logistique dans un Pipeline
# Le RobustScaler est refait DANS le Pipeline, à chaque pli.
# >>> NOUVEAU (notebook) : C_OPT est une variable globale.
#     Valeur de départ = params.yaml ; remplacée par le résultat du GridSearchCV.
# ============================================================
C_OPT = float(P["model"]["C"])


def construire_modele(colonnes, class_weight=None, C=None):
    if C is None:
        C = C_OPT
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
# 12b. >>> NOUVEAU (notebook) : GRIDSEARCHCV pour choisir C
# Protocole : Leave-One-Edition-Out (on cache une édition à chaque pli)
# Critère : Top-1 le plus haut, puis rang moyen le plus bas, puis plus petit C.
# Pour désactiver : grid_search: false dans params.yaml (C fixe).
# ============================================================
def score_top1(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    return float(np.asarray(y)[np.argmax(proba)] == 1)


def score_rang(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    y_arr = np.asarray(y)
    if y_arr.sum() == 0:
        return -float(len(y_arr))
    g = int(np.argmax(y_arr))
    return -float((proba >= proba[g]).sum())      # -rang du vrai gagnant


grid_search_actif = bool(P["model"].get("grid_search", True))

if grid_search_actif:
    grille = P["model"].get("grille_C",
                            [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    cols_gs = jeux_features[P["model"]["jeu_features"]]
    cw_gs = P["model"]["class_weight"]

    gs = GridSearchCV(
        estimator=construire_modele(cols_gs, cw_gs, C=1.0),
        param_grid={"lr__C": grille},
        scoring={"top1": score_top1, "neg_rang": score_rang},
        refit=False,
        cv=LeaveOneGroupOut(),
        n_jobs=1,
    )
    gs.fit(X_all[cols_gs], y_all, groups=groupes_all)

    res_gs = pd.DataFrame({
        "C": [p["lr__C"] for p in gs.cv_results_["params"]],
        "Top-1": gs.cv_results_["mean_test_top1"],
        "Rang moyen": -gs.cv_results_["mean_test_neg_rang"],
    }).sort_values(["Top-1", "Rang moyen", "C"],
                   ascending=[False, True, True]).reset_index(drop=True)

    C_OPT = float(res_gs.loc[0, "C"])

    print("\n===== GRIDSEARCHCV : CHOIX DE C =====")
    print(res_gs.round(3).to_string(index=False))
    best = res_gs.loc[0]
    egalite_top1 = res_gs[res_gs["Top-1"] == best["Top-1"]]
    egalite_rang = egalite_top1[egalite_top1["Rang moyen"] == best["Rang moyen"]]
    print(f"\nMeilleur C = {C_OPT}")
    print(f"• Top-1 maximal : {best['Top-1']:.3f} "
          f"({best['Top-1'] * groupes_all.nunique():.0f} éditions sur {groupes_all.nunique()})")
    print(f"• C avec ce Top-1 : {sorted(egalite_top1['C'].tolist())}")
    if len(egalite_top1) > 1:
        print(f"• Après départage par le rang moyen : {sorted(egalite_rang['C'].tolist())}")
    if len(egalite_rang) > 1:
        print(f"• Dernier critère : plus petit C -> C = {C_OPT}")
    if C_OPT <= 0.003:
        print("⚠️ C très petit : risque d'underfitting.")
    elif C_OPT >= 30:
        print("⚠️ C très grand : risque d'overfitting.")
    else:
        print("✅ C dans une zone intermédiaire : bon compromis.")
else:
    print(f"\nGridSearchCV désactivé : C fixé par params.yaml = {C_OPT}")

# ============================================================
# 13. ÉVALUATION PAR ÉDITION (Leave-One-Edition-Out)
# On cache une édition, on entraîne sur les autres, on regarde le rang du vrai gagnant.
# Ces chiffres sont HONNÊTES : le modèle n'a jamais vu l'édition évaluée.
# ============================================================
def evaluer_par_edition(colonnes, class_weight):
    Xc = X_all[colonnes]
    modele = construire_modele(colonnes, class_weight)     # utilise C_OPT
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
            "predit": data.loc[proba.loc[idx].idxmax(), "joueur"],   # AJOUT : le 1er du classement
            "rang": int((proba.loc[idx] >= proba.loc[g]).sum()),
            "part_gagnant": part.loc[g],
        })
    det = pd.DataFrame(lignes).set_index("edition")

    # 1 annonce (le 1er) et 1 vrai gagnant par édition : Precision = Recall = F1 = Top-1
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
print(f"\n===== BASELINE BRUTE vs PRÉTRAITÉ vs PRÉTRAITÉ + FE (C = {C_OPT}) =====")
print(df_res.round(3).to_string())

meilleur_auto = df_res.index[0]
print(f"\nPremière ligne du tableau (tri automatique) : {meilleur_auto}")

# Choix manuel (params.yaml) : Prétraité et Baseline brute sont à égalité,
# on retient Prétraité pour sa robustesse aux valeurs extrêmes.
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

print("\n===== MODÈLE FINAL UTILISÉ POUR 2026 =====")
print(f"Jeu de variables    : {nom_jeu}")
print(f"class_weight        : {cw}")
print(f"C                   : {C_OPT}")
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

# Graphique du Top 10 (visible seulement avec SHOW_PLOTS=1)
top10 = classement.head(10)
plt.figure(figsize=(9, 5))
sns.barplot(data=top10, y="joueur", x="chance_%", color="steelblue")
plt.title("Top 10 candidats Ballon d'Or 2026 (chance %)")
plt.xlabel("Chance (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()
plt.close("all")

# ============================================================
# >>> MLOPS (ajout 4) : 15. SAUVEGARDE + SUIVI MLFLOW
# ============================================================

# --- 15.1 Les métriques de la configuration retenue -------------------
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
bundle = {
    "model": modele_final,
    "ohe": ohe,
    "yeo": yeo,
    "iqr_limits": iqr_limits,
    "features": colonnes_finales,
    "cols_relatives_src": cols_relatives_src,
    "C": C_OPT,
}

# --- 15.3 Sauvegarde sur le disque -------------------------------------
os.makedirs(BASE_DIR / "models", exist_ok=True)
os.makedirs(BASE_DIR / "metrics", exist_ok=True)

chemin_backtest = BASE_DIR / P["paths"].get("backtest", "metrics/backtest.json")
chemin_classement = BASE_DIR / P["paths"].get("classement", "metrics/classement_2026.json")

joblib.dump(bundle, BASE_DIR / P["paths"]["model"])               # -> models/model.pkl
with open(BASE_DIR / P["paths"]["metrics"], "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)                               # -> metrics/metrics.json

# >>> NOUVEAU : ce que le dashboard affiche (source unique de vérité)
det = details[meilleur]
backtest = {
    "config": {
        "jeu_features": nom_jeu,
        "class_weight": str(cw),
        "C": C_OPT,
        "grid_search": grid_search_actif,
        "n_features": len(colonnes_finales),
        "n_editions": int(groupes_all.nunique()),
        "n_candidats": int(X_all.shape[0]),
    },
    "summary": metrics,
    "per_edition": [
        {
            "edition": int(ed),
            "gagnant": str(row["gagnant"]),
            "predit": str(row["predit"]),
            "rang": int(row["rang"]),
            "part_gagnant_pct": float(100 * row["part_gagnant"]),
        }
        for ed, row in det.iterrows()
    ],
}
with open(chemin_backtest, "w", encoding="utf-8") as f:
    json.dump(backtest, f, indent=2, ensure_ascii=False)          # -> metrics/backtest.json

with open(chemin_classement, "w", encoding="utf-8") as f:
    json.dump([
        {"rang": int(l["rang"]), "joueur": str(l["joueur"]),
         "chance_pct": float(l["chance_%"]), "proba_brute": float(l["proba_brute"])}
        for _, l in classement.iterrows()
    ], f, indent=2, ensure_ascii=False)                           # -> metrics/classement_2026.json

# --- 15.4 MLflow : le cahier de laboratoire ------------------------------
mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment(P["mlflow"]["experiment"])

with mlflow.start_run():
    mlflow.log_params({
        "C": C_OPT,
        "grid_search": grid_search_actif,
        "class_weight": str(cw),
        "jeu_features": nom_jeu,
        "nb_features": len(colonnes_finales),
        "test_size": P["split"]["test_size"],
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(modele_final, name="model",
                             serialization_format="cloudpickle")
    mlflow.log_artifact(str(BASE_DIR / P["paths"]["model"]))
    mlflow.log_artifact(str(BASE_DIR / P["paths"]["metrics"]))
    mlflow.log_artifact(str(chemin_backtest))
    mlflow.log_artifact(str(chemin_classement))

print("\n✅ Modèle sauvegardé, métriques enregistrées dans MLflow")
print(metrics)