
---

## 🧠 Pipeline Machine Learning

Le projet suit une **méthodologie ML rigoureuse** :

1. Chargement des données  
2. Analyse exploratoire des données (EDA)  
3. Préparation des données  
4. Séparation Train / Validation / Test (70 / 15 / 15)  
5. Modèle baseline  
6. Sélection du modèle par validation croisée  
7. Prétraitement des données (standardisation)  
8. Entraînement du modèle final  
9. Évaluation sur l’ensemble de validation  
10. Évaluation finale sur l’ensemble de test  
11. Conclusion  

Toutes les étapes sont documentées dans un **notebook Jupyter**,
tandis que la logique métier est organisée dans le dossier `src/`.

---

## 📊 Modèles utilisés

- **Régression linéaire**
- **Ridge Regression** (testée via validation croisée)

Après comparaison, la **régression linéaire** a été retenue comme modèle final,
en raison de ses excellentes performances et de sa simplicité.

---

## 📈 Métriques d’évaluation

Les performances du modèle sont évaluées à l’aide des métriques suivantes :

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient de détermination (R²)
- Erreur relative

### 🔍 Performance finale sur le jeu de test (exemple)

- **R²** ≈ 0.999999  
- **Erreur relative** < 0.1 %

Ces résultats montrent une **très forte capacité de généralisation**
sur des données jamais vues.

---

## 🧪 Modèle baseline

Un modèle baseline consistant à prédire **la moyenne des prix**
est utilisé comme référence minimale.

Le modèle final surpasse largement cette baseline,
confirmant la pertinence des features sélectionnées
et du pipeline de prétraitement.

---

## ⚠️ Remarques et limites

- Le jeu de données présente une **relation linéaire très forte**
entre la variable cible et certaines features (ex. surface).
- Cela explique les scores R² extrêmement élevés.
- Les performances peuvent être différentes sur des jeux de données plus complexes.

---

## 🚀 Améliorations possibles

- Tester des modèles non linéaires (Random Forest, XGBoost)
- Effectuer une sélection de variables plus poussée
- Sauvegarder et déployer le modèle
- Appliquer la pipeline à un autre jeu de données immobilier

---

## 🛠️ Prérequis

Installation des dépendances :

```bash
pip install -r requirements.txt
