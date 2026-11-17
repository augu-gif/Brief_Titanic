# Brief Titanic

> **Quels facteurs ont réellement influencé les chances de survie des passagers du Titanic ?**

Projet réalisé dans le cadre de la formation Simplon : en tant que Data Scientist junior, analyser les données des passagers pour un **musée maritime** qui prépare une exposition sur le naufrage.

## Sommaire

- [Données](#données)
- [Démarche](#démarche)
- [Résultats](#résultats)
- [Structure du dépôt](#structure-du-dépôt)
- [Relancer le projet](#relancer-le-projet)
- [Critères du brief](#critères-du-brief)

## Données

Dataset Titanic ([Kaggle](https://www.kaggle.com/c/titanic)) : **891 passagers × 12 colonnes**, cible `Survived` (1 = a survécu, 38 % des passagers).

| Problème | Traitement |
|---|---|
| `Age` manquant (20 %) | Imputé par l'**âge médian du titre** (un `Master` reçoit 3,5 ans, un `Mr` 30 ans) |
| `Cabin` manquante (77 %) | Supprimée, remplacée par `Has_cabin` (1 si la cabine est connue) |
| `Embarked` manquant (2 lignes) | Imputé par le port le plus fréquent (`S`, Southampton) |
| Titre caché dans `Name` | Extrait (`Mr`, `Mrs`, `Miss`, `Master`, `Autres`), puis `Name` supprimée |
| Colonnes sans valeur prédictive | `PassengerId` et `Ticket` supprimées |
| Variables texte | `Sex` → 0/1, `Embarked` et `Title` → encodage one-hot |

Le dataset final (`Dataset/df_propre.csv`) ne contient **aucune valeur manquante** et **uniquement des variables numériques**.

## Démarche

1. **Analyse exploratoire** : structure, valeurs manquantes, distributions, corrélations, puis hypothèses (le sexe, la classe et l'âge comptent ; le port d'embarquement n'a pas d'effet direct).
2. **Nettoyage et transformation** : traitements ci-dessus, chacun justifié dans le notebook.
3. **Feature engineering** : `Title`, `Has_cabin`, `FamilySize` (taille de la famille), `IsAlone` (voyage seul), `LargeFamily` (5 personnes ou plus).
4. **Modélisation** : 3 modèles comparés (régression logistique, Random Forest, XGBoost) sur le même jeu de test (20 % des passagers, `random_state=42`).

## Résultats

| Modèle | Accuracy (validation croisée) | Accuracy (test) |
|---|---|---|
| Baseline (prédit toujours « décédé ») | 61,6 % | 61,5 % |
| XGBoost (réglages par défaut) | 79,4 % | 76,5 % |
| Random Forest | – | 79,3 % |
| **Régression logistique** | **82,8 %** | **83,2 %** |

La **régression logistique** obtient les meilleurs résultats stables, sans surapprentissage, et c'est le modèle le plus facile à expliquer au musée.

**Ce qui a compté pour survivre :**

1. **Être une femme ou un enfant** : 74 % de survie chez les femmes contre 19 % chez les hommes (16 % pour les hommes adultes, `Mr`).
2. **La classe** : 63 % de survie en 1re classe contre 24 % en 3e.
3. **Les très grandes familles** (5 personnes ou plus) ont été fortement pénalisées.
4. **L'âge** : à profil égal, les passagers plus âgés ont moins survécu.
5. Le **prix du billet** et le **port d'embarquement** n'ont qu'un rôle secondaire une fois la classe prise en compte.

> Ce sont des **associations** observées dans les données, pas des causes prouvées.

## Structure du dépôt

```
Brief_Titanic/
├── Dataset/
│   ├── dataset-titanic-6afac2fa9ddd8043393546.csv   # données brutes fournies par le musée
│   └── df_propre.csv                                # dataset nettoyé et encodé
├── Notebooks/
│   └── Brief_Titanic.ipynb                          # EDA, nettoyage, modélisation et conclusion
├── Veille/                                          # veille : classification, nettoyage, pipeline ML
└── README.md
```

## Relancer le projet

Le notebook est prévu pour **Google Colab** :

1. Ouvrir `Notebooks/Brief_Titanic.ipynb` dans [Google Colab](https://colab.research.google.com/).
2. Lancer **Exécution → Tout exécuter**.
3. À la première cellule, importer le fichier `Dataset/dataset-titanic-6afac2fa9ddd8043393546.csv` quand Colab le demande.

Bibliothèques utilisées (déjà installées sur Colab) : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.

> **En local** (VS Code, Jupyter) : installer les bibliothèques avec `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`, supprimer la première cellule (`google.colab`) et remplacer le chemin `/content/...` par `../Dataset/dataset-titanic-6afac2fa9ddd8043393546.csv`.

Toutes les étapes aléatoires utilisent `random_state=42` : les résultats sont reproductibles.

## Critères du brief

- [x] Valeurs manquantes traitées, chaque choix justifié dans le notebook
- [x] Variables catégorielles encodées, dataset final entièrement numérique
- [x] Au moins 2 nouvelles variables créées (`Title`, `Has_cabin`, `FamilySize`, `IsAlone`, `LargeFamily`)
- [x] Au moins 3 modèles de classification comparés
- [x] Notebook exécuté de bout en bout sans erreur, résultats reproductibles (`random_state` fixé)
- [x] README permettant de comprendre et de relancer le projet

**Compétences visées :** C34 (nettoyage, transformation et réduction de dimensionnalité des datasets), C39 (analyse exploratoire des données).
