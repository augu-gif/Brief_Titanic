# Veille technologique : la classification en Python

> Synthèse sur les méthodes, les outils et les tendances de la classification supervisée en Python.

---

## 1. Définition

La **classification** est une tâche d'**apprentissage supervisé** : à partir d'exemples étiquetés, le modèle apprend à prédire une **catégorie discrète** pour de nouvelles données.

Exemples d'usage :
- détection de spam (spam / non spam) ;
- diagnostic médical (malade / sain) ;
- analyse de sentiment (positif / neutre / négatif) ;
- reconnaissance d'images (chat, chien, oiseau…) ;
- détection de fraude bancaire.

### Les types de classification

| Type | Description | Exemple |
|---|---|---|
| **Binaire** | 2 classes possibles | Spam / non spam |
| **Multiclasse** | Plus de 2 classes, une seule par exemple | Espèce d'iris (setosa, versicolor, virginica) |
| **Multilabel** | Plusieurs étiquettes possibles pour un même exemple | Tags d'un article (sport, politique, économie) |

---

## 2. Les algorithmes classiques

| Algorithme | Principe | Points forts | Limites |
|---|---|---|---|
| **Régression logistique** | Modèle linéaire + fonction sigmoïde | Simple, rapide, interprétable, bonne référence de départ | Frontière de décision linéaire |
| **k plus proches voisins (k-NN)** | Vote des k exemples les plus proches | Intuitif, aucun entraînement | Lent sur gros volumes, sensible à l'échelle des données |
| **Naive Bayes** | Théorème de Bayes avec hypothèse d'indépendance | Très rapide, efficace sur le texte | Hypothèse d'indépendance rarement vraie |
| **SVM** | Hyperplan qui maximise la marge entre classes | Efficace en grande dimension, noyaux non linéaires | Lent sur de gros jeux de données |
| **Arbre de décision** | Suite de règles « si… alors… » | Lisible, explicable | Surapprentissage facile |
| **Random Forest** | Ensemble d'arbres (bagging) | Robuste, peu de réglages | Moins interprétable, plus lourd |
| **Gradient Boosting** | Arbres construits successivement pour corriger les erreurs | Souvent le plus performant sur données tabulaires | Nombreux hyperparamètres |
| **Réseaux de neurones** | Couches de neurones apprenant des représentations | Indispensables pour images, son, texte | Besoin de beaucoup de données et de calcul |

---

## 3. L'écosystème Python

### Bibliothèques principales

| Bibliothèque | Usage |
|---|---|
| [scikit-learn](https://scikit-learn.org/) | La référence du machine learning classique : modèles, `Pipeline`, `ColumnTransformer`, `GridSearchCV`, métriques |
| [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/) | Gradient boosting performant, très utilisé en entreprise et en compétition |
| [PyTorch](https://pytorch.org/) / [Keras](https://keras.io/) | Deep learning (images, son, texte) |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Modèles pré-entraînés pour la classification de texte (BERT, CamemBERT pour le français…) |

### Outils complémentaires

| Outil | Usage |
|---|---|
| [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) | Manipulation des données |
| [imbalanced-learn](https://imbalanced-learn.org/) | Gestion des classes déséquilibrées (SMOTE, sous-échantillonnage) |
| [SHAP](https://shap.readthedocs.io/) / [LIME](https://github.com/marcotcr/lime) | Explicabilité des prédictions |
| [Optuna](https://optuna.org/) | Optimisation des hyperparamètres |
| [MLflow](https://mlflow.org/) | Suivi des expériences et des modèles |
| [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) | Visualisation (matrices de confusion, courbes ROC) |

---

## 4. Évaluer un modèle de classification

### La matrice de confusion

|  | Prédit positif | Prédit négatif |
|---|---|---|
| **Réellement positif** | Vrai positif (VP) | Faux négatif (FN) |
| **Réellement négatif** | Faux positif (FP) | Vrai négatif (VN) |

### Les métriques

| Métrique | Formule | Question à laquelle elle répond |
|---|---|---|
| **Accuracy** | (VP + VN) / total | Quelle part des prédictions est correcte ? *Trompeuse si les classes sont déséquilibrées.* |
| **Precision** | VP / (VP + FP) | Quand le modèle dit « positif », a-t-il raison ? |
| **Recall** (rappel) | VP / (VP + FN) | Parmi les vrais positifs, combien sont détectés ? |
| **F1-score** | 2 × (P × R) / (P + R) | Compromis entre precision et recall |
| **ROC-AUC** | Aire sous la courbe ROC | Capacité à séparer les classes, quel que soit le seuil |
| **PR-AUC** | Aire sous la courbe precision-recall | Plus parlante que la ROC-AUC quand les classes sont déséquilibrées |

### La validation croisée

Plutôt qu'un seul découpage train/test, on utilise `StratifiedKFold` : le jeu de données est découpé en *k* parties en conservant la proportion de chaque classe, et le modèle est évalué *k* fois.

---

## 5. Workflow type avec scikit-learn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. Charger les données
X, y = load_breast_cancer(return_X_y=True)

# 2. Séparer entraînement et test (stratify conserve la proportion des classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Comparer plusieurs modèles en validation croisée
modeles = {
    "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for nom, modele in modeles.items():
    scores = cross_val_score(modele, X_train, y_train, cv=cv, scoring="f1")
    print(f"{nom} : F1 = {scores.mean():.3f} ± {scores.std():.3f}")

# 4. Entraîner le modèle retenu et l'évaluer sur le jeu de test
meilleur = modeles["Random Forest"].fit(X_train, y_train)
y_pred = meilleur.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

### Les étapes clés

1. **Explorer les données** : valeurs manquantes, distribution des classes, types de variables.
2. **Préparer** : encodage des variables catégorielles, mise à l'échelle, gestion des valeurs manquantes (dans un `Pipeline` pour éviter les fuites de données).
3. **Établir une référence** : un modèle simple (régression logistique, `DummyClassifier`).
4. **Comparer** plusieurs modèles en validation croisée.
5. **Optimiser** les hyperparamètres (`GridSearchCV`, `RandomizedSearchCV`, Optuna).
6. **Évaluer** une seule fois sur le jeu de test.
7. **Expliquer** les prédictions (importance des variables, SHAP).

---

## 6. Problèmes fréquents et solutions

| Problème | Symptôme | Solutions |
|---|---|---|
| **Classes déséquilibrées** | Bonne accuracy mais la classe minoritaire n'est jamais prédite | `class_weight="balanced"`, SMOTE, métriques adaptées (F1, PR-AUC), ajustement du seuil |
| **Surapprentissage** | Excellent score en entraînement, mauvais en test | Régularisation, validation croisée, plus de données, modèle plus simple |
| **Fuite de données** | Scores trop beaux pour être vrais | Prétraitement à l'intérieur d'un `Pipeline`, séparation train/test faite en premier |
| **Variables à échelles différentes** | k-NN, SVM ou régression logistique peu performants | `StandardScaler`, `MinMaxScaler` |
| **Probabilités mal calibrées** | Un score de 0,9 ne correspond pas à 90 % de chances | `CalibratedClassifierCV` |

---

## 7. Tendances actuelles

- **Modèles de fondation pour les données tabulaires** : des modèles comme [TabPFN](https://github.com/PriorLabs/TabPFN) sont pré-entraînés et classent de petits jeux de données sans entraînement classique.
- **LLM et classification de texte** : classification *zero-shot* ou *few-shot* sans données annotées, ou **embeddings + classifieur simple** (régression logistique sur des vecteurs de phrases).
- **AutoML** : [AutoGluon](https://auto.gluon.ai/) et [FLAML](https://microsoft.github.io/FLAML/) testent et combinent automatiquement plusieurs modèles.
- **IA explicable et responsable** : explicabilité, détection des biais, équité entre groupes, en lien avec les obligations de l'**AI Act** européen.
- **MLOps** : suivi des expériences, versionnement des modèles, surveillance de la dérive des données en production.

---

## 8. Sources de veille

| Source | Type |
|---|---|
| [Documentation scikit-learn](https://scikit-learn.org/stable/user_guide.html) et ses [notes de version](https://scikit-learn.org/stable/whats_new.html) | Documentation officielle |
| [Blog Hugging Face](https://huggingface.co/blog) | Articles, nouveaux modèles |
| [Kaggle](https://www.kaggle.com/) | Compétitions, notebooks, jeux de données |
| [Papers with Code](https://paperswithcode.com/) | Articles de recherche avec code |
| [Towards Data Science](https://towardsdatascience.com/) | Articles de vulgarisation |
| [Machine Learning Mastery](https://machinelearningmastery.com/) | Tutoriels pratiques |

---

## 9. Conclusion

- Pour des **données tabulaires**, la combinaison scikit-learn + gradient boosting (XGBoost, LightGBM, CatBoost) reste la plus efficace.
- Pour le **texte et les images**, les modèles pré-entraînés (Hugging Face, PyTorch) dominent.
- Le choix de la **métrique** compte autant que celui de l'algorithme, surtout quand les classes sont déséquilibrées.
- Les tendances vont vers l'**automatisation** (AutoML, modèles de fondation) et l'**explicabilité**, poussée par la réglementation.

---

## 10. Lexique

Les exemples s'appuient sur le jeu de données **Titanic**, où l'on cherche à prédire si un passager a survécu.

### Les données

| Terme | Explication | Exemple Titanic |
|---|---|---|
| **Jeu de données** (*dataset*) | Le tableau de données utilisé : une ligne par exemple, une colonne par information. | Le fichier `train.csv`, une ligne par passager. |
| **Variable** (*feature*) | Une colonne utilisée pour faire la prédiction. On la note souvent `X`. | `Age`, `Sex`, `Pclass`, `Fare` |
| **Cible** (*target*, *label*, étiquette) | Ce que l'on veut prédire. On la note souvent `y`. | `Survived` (0 ou 1) |
| **Classe** | Une des valeurs possibles de la cible. | « survécu » et « décédé » |
| **Variable numérique** | Variable qui contient un nombre. | `Age`, `Fare` |
| **Variable catégorielle** | Variable qui contient une catégorie (du texte ou un code). | `Sex`, `Embarked` (C, Q, S) |
| **Valeur manquante** | Case vide dans le tableau (`NaN` dans pandas). | Environ 20 % des `Age` sont manquants. |
| **Jeu d'entraînement** (*train*) | Partie des données sur laquelle le modèle apprend. | 80 % des passagers |
| **Jeu de test** (*test*) | Partie mise de côté pour vérifier le modèle sur des données qu'il n'a jamais vues. | Les 20 % restants |
| **Stratification** (`stratify=y`) | Découper les données en gardant la même proportion de chaque classe dans chaque partie. | Environ 38 % de survivants dans le train **et** dans le test |

### Le modèle et l'apprentissage

| Terme | Explication |
|---|---|
| **Apprentissage supervisé** | Le modèle apprend à partir d'exemples dont on connaît déjà la réponse (la cible). |
| **Modèle** | Le programme qui, à partir des variables, produit une prédiction. |
| **Entraînement** (`fit`) | Étape où le modèle apprend à partir du jeu d'entraînement. |
| **Prédiction** (`predict`) | Étape où le modèle donne sa réponse pour de nouvelles données. |
| **Probabilité** (`predict_proba`) | Le modèle donne un score entre 0 et 1 au lieu d'une réponse tranchée. Ex. : « 0,82 de chances d'avoir survécu ». |
| **Seuil de décision** | Valeur à partir de laquelle une probabilité devient une prédiction « positive ». Par défaut 0,5 ; on peut le changer pour privilégier la precision ou le recall. |
| **Frontière de décision** | La « ligne » (ou surface) qui sépare les classes dans l'espace des variables. Elle est droite pour un modèle linéaire, courbe pour les autres. |
| **Référence** (*baseline*) | Un modèle très simple qui sert de point de comparaison. `DummyClassifier` prédit toujours la classe majoritaire : sur le Titanic, il obtient environ 62 % d'accuracy en prédisant « décédé » pour tout le monde. Un vrai modèle doit faire mieux. |

### Paramètres et hyperparamètres

C'est une distinction essentielle :

| | **Paramètre** | **Hyperparamètre** |
|---|---|---|
| **Qui le fixe ?** | Le modèle, **pendant** l'entraînement | Toi, **avant** l'entraînement |
| **Comment ?** | Appris automatiquement à partir des données | Choisi à la main ou par une recherche automatique |
| **Exemple** | Les coefficients d'une régression logistique (le « poids » de `Sex` ou de `Age`) | Le nombre d'arbres d'une Random Forest, leur profondeur maximale |

**Analogie** : pour un élève qui révise, les **paramètres** sont ce qu'il retient, et les **hyperparamètres** sont l'organisation de ses révisions (nombre d'heures, nombre d'exercices, méthode). On règle l'organisation, puis l'élève apprend.

En scikit-learn, les hyperparamètres sont les arguments passés à la création du modèle :

```python
from sklearn.ensemble import RandomForestClassifier

# n_estimators et max_depth sont des hyperparamètres : on les choisit
modele = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

# Pendant fit, le modèle apprend ses paramètres (les règles de chaque arbre)
modele.fit(X_train, y_train)
```

#### Hyperparamètres courants

| Algorithme | Hyperparamètre | Rôle |
|---|---|---|
| Régression logistique | `C` | Force de la régularisation : plus `C` est petit, plus le modèle est simple. |
| k-NN | `n_neighbors` | Nombre de voisins consultés pour voter. |
| Arbre de décision | `max_depth` | Profondeur maximale de l'arbre : trop grande, il apprend par cœur. |
| Arbre de décision | `min_samples_leaf` | Nombre minimum d'exemples dans chaque feuille. |
| Random Forest | `n_estimators` | Nombre d'arbres dans la forêt. |
| SVM | `C`, `kernel`, `gamma` | Tolérance aux erreurs, type de frontière (linéaire ou courbe), influence de chaque point. |
| Gradient Boosting | `learning_rate` | Taille de la correction apportée par chaque nouvel arbre. |
| Réseau de neurones | nombre de couches, `epochs`, `batch_size` | Taille du réseau, nombre de passages sur les données, taille des paquets d'exemples. |

#### Comment choisir les hyperparamètres ?

| Méthode | Principe |
|---|---|
| **Recherche en grille** (`GridSearchCV`) | On liste des valeurs à tester et on essaie **toutes** les combinaisons. Exhaustif mais lent. |
| **Recherche aléatoire** (`RandomizedSearchCV`) | On tire au hasard un nombre fixé de combinaisons. Plus rapide, souvent presque aussi bon. |
| **Optimisation bayésienne** (Optuna) | Chaque essai tient compte des résultats précédents pour viser les zones prometteuses. |

```python
from sklearn.model_selection import GridSearchCV

grille = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 10, None],
}
recherche = GridSearchCV(RandomForestClassifier(random_state=42), grille, cv=5, scoring="f1")
recherche.fit(X_train, y_train)
print(recherche.best_params_)  # meilleure combinaison trouvée
```

Ici, 3 × 4 = 12 combinaisons, chacune évaluée 5 fois (`cv=5`) : 60 entraînements au total.

### Évaluation

| Terme | Explication |
|---|---|
| **Métrique** | Un nombre qui mesure la qualité des prédictions (accuracy, F1-score…). Voir la [section 4](#4-évaluer-un-modèle-de-classification). |
| **Vrai positif / Faux positif** | Sur le Titanic, en prenant « survécu » comme classe positive : un **vrai positif** est un survivant prédit survivant ; un **faux positif** est un passager décédé prédit survivant. |
| **Vrai négatif / Faux négatif** | Un **vrai négatif** est un passager décédé prédit décédé ; un **faux négatif** est un survivant prédit décédé. |
| **Courbe ROC** | Courbe qui montre, pour tous les seuils possibles, le taux de vrais positifs en fonction du taux de faux positifs. Plus elle monte vite, meilleur est le modèle. |
| **AUC** (*Area Under the Curve*) | Aire sous la courbe : 0,5 = hasard, 1 = parfait. |
| **Validation croisée** (*cross-validation*) | On découpe le train en *k* parts (*folds*) ; le modèle est entraîné *k* fois, en gardant à chaque fois une part différente pour l'évaluer. On obtient une moyenne plus fiable qu'avec un seul découpage. |
| **Fold** | Une des *k* parts de la validation croisée. |
| **`random_state`** | « Graine » du hasard : la fixer (ex. `42`) garantit d'obtenir les mêmes résultats à chaque exécution. |

### Les problèmes d'apprentissage

| Terme | Explication |
|---|---|
| **Surapprentissage** (*overfitting*) | Le modèle apprend les données d'entraînement **par cœur**, bruit compris, et se trompe sur de nouvelles données. Signe : très bon score en train, mauvais en test. |
| **Sous-apprentissage** (*underfitting*) | Le modèle est trop simple pour saisir la logique des données. Signe : mauvais score en train **et** en test. |
| **Régularisation** | Pénalité ajoutée pendant l'entraînement pour empêcher le modèle de devenir trop complexe. Limite le surapprentissage. |
| **Fuite de données** (*data leakage*) | Le modèle a eu accès, sans qu'on le veuille, à des informations du jeu de test ou à la réponse. Ses scores sont alors faussement bons. Ex. : calculer l'âge moyen sur **toutes** les données (test compris) pour remplir les âges manquants. |
| **Classes déséquilibrées** | Une classe est beaucoup plus fréquente que l'autre. Le modèle a tendance à ignorer la classe rare. |
| **Calibration** | Le fait que les probabilités prédites correspondent à la réalité : parmi les passagers prédits à 0,8, environ 80 % doivent avoir survécu. |

### La préparation des données

| Terme | Explication | Exemple Titanic |
|---|---|---|
| **Prétraitement** (*preprocessing*) | Toutes les transformations appliquées aux données avant l'entraînement. | Remplir les âges, encoder le sexe… |
| **Imputation** | Remplacer les valeurs manquantes (par la moyenne, la médiane, la valeur la plus fréquente…). `SimpleImputer` en scikit-learn. | Remplacer les `Age` manquants par la médiane. |
| **Encodage** | Transformer une variable catégorielle en nombres, car les modèles ne comprennent que les nombres. | `Sex` : `male` → 0, `female` → 1 |
| **One-hot encoding** | Créer une colonne 0/1 par catégorie. `OneHotEncoder` en scikit-learn. | `Embarked` devient `Embarked_C`, `Embarked_Q`, `Embarked_S` |
| **Mise à l'échelle** (*scaling*) | Ramener les variables numériques à des ordres de grandeur comparables. | `Fare` (0 à 512) et `Age` (0 à 80) ramenés autour de 0 |
| **Standardisation** (`StandardScaler`) | Mise à l'échelle qui donne à chaque variable une moyenne de 0 et un écart-type de 1. | |
| **Feature engineering** | Créer de nouvelles variables utiles à partir des existantes. | `FamilySize = SibSp + Parch + 1`, ou le titre extrait du nom (`Mr`, `Mrs`, `Miss`) |
| **Pipeline** | Enchaînement automatique des étapes (prétraitement puis modèle) dans un seul objet. Évite les fuites de données et simplifie le code. | |
| **`ColumnTransformer`** | Applique des traitements différents selon les colonnes. | Imputation + standardisation pour `Age` et `Fare`, one-hot pour `Sex` et `Embarked` |
| **SMOTE** | Technique qui crée des exemples artificiels de la classe rare pour rééquilibrer les classes. | |
| **`class_weight="balanced"`** | Option qui donne plus d'importance aux erreurs sur la classe rare pendant l'entraînement. | |

### Les familles d'algorithmes

| Terme | Explication |
|---|---|
| **Modèle linéaire** | Modèle dont la frontière de décision est une droite (ou un plan). Ex. : régression logistique. |
| **Sigmoïde** | Fonction en forme de « S » qui transforme n'importe quel nombre en une valeur entre 0 et 1, interprétée comme une probabilité. Utilisée par la régression logistique. |
| **Marge** (SVM) | Distance entre la frontière de décision et les points les plus proches. Le SVM cherche la marge la plus large. |
| **Noyau** (*kernel*, SVM) | Astuce mathématique qui permet au SVM de tracer des frontières courbes. |
| **Méthode d'ensemble** | Combiner plusieurs modèles pour obtenir une meilleure prédiction que chacun séparément. |
| **Bagging** | Méthode d'ensemble où plusieurs modèles sont entraînés **en parallèle** sur des échantillons différents, puis votent. Ex. : Random Forest. |
| **Boosting** | Méthode d'ensemble où les modèles sont entraînés **l'un après l'autre**, chacun corrigeant les erreurs du précédent. Ex. : XGBoost, LightGBM. |
| **Gradient** | Indique dans quelle direction ajuster le modèle pour réduire ses erreurs. Le « gradient boosting » l'utilise pour guider chaque nouvel arbre. |
| **Deep learning** | Apprentissage avec des réseaux de neurones à nombreuses couches. |

### Les tendances

| Terme | Explication |
|---|---|
| **Modèle pré-entraîné** | Modèle déjà entraîné sur une énorme quantité de données, que l'on réutilise. |
| **Fine-tuning** | Réentraîner légèrement un modèle pré-entraîné sur ses propres données. |
| **Modèle de fondation** | Très grand modèle pré-entraîné, réutilisable pour de nombreuses tâches (ex. les LLM, TabPFN). |
| **LLM** (*Large Language Model*) | Grand modèle de langage, comme ceux derrière les assistants conversationnels. |
| **Zero-shot / Few-shot** | Classer sans aucun exemple annoté (*zero-shot*) ou avec seulement quelques exemples (*few-shot*). |
| **Embedding** | Représentation d'un texte (ou d'une image) sous forme d'une liste de nombres qui capture son sens. Deux textes proches ont des embeddings proches. |
| **AutoML** | Outils qui testent et règlent automatiquement modèles et hyperparamètres. |
| **Explicabilité** | Capacité à comprendre **pourquoi** un modèle a fait une prédiction. Ex. : SHAP montre que `Sex` et `Pclass` pèsent le plus sur la survie. |
| **MLOps** | Pratiques pour déployer, suivre et maintenir des modèles en production. |
| **Dérive des données** (*data drift*) | Les données réelles changent avec le temps et ne ressemblent plus aux données d'entraînement, ce qui dégrade le modèle. |
