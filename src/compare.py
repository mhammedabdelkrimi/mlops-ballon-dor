import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
nouveau = json.load(open(BASE / "metrics" / "metrics.json", encoding="utf-8"))
chemin_prod = BASE / "metrics" / "production.json"

if chemin_prod.exists():
    prod_f1 = json.load(open(chemin_prod, encoding="utf-8"))["f1"]
else:
    prod_f1 = None                     # aucun modèle en production : on déploie

promote = prod_f1 is None or nouveau["f1"] > prod_f1
print(f"Nouveau F1={nouveau['f1']:.4f} | Production F1={prod_f1} | Déployer={promote}")

with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as f:
    f.write(f"promote={'true' if promote else 'false'}\n")

if promote:
    json.dump({"f1": nouveau["f1"], "rang_moyen": nouveau["rang_moyen"]},
              open(chemin_prod, "w", encoding="utf-8"), indent=2)