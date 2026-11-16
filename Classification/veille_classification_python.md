# Veille technologique : la classification en Python

## 1. Définition

La **classification** est une tâche d'**apprentissage supervisé** : à partir d'exemples dont on connaît la réponse, un modèle apprend à prédire une **catégorie** pour de nouvelles données (ex. : reconnaître l'espèce d'une fleur à partir de la taille de ses pétales).

Elle peut être **binaire** (2 classes), **multiclasse** (plus de 2 classes) ou **multilabel** (plusieurs étiquettes par exemple).

**Vocabulaire de base :**

- **Variables** (*features*, `X`) : les colonnes utilisées pour prédire.
- **Cible** (*target*, `y`) : ce que l'on veut prédire.
- **Jeu d'entraînement / de test** : les données sur lesquelles le modèle apprend, et celles, mises de côté, qui servent à vérifier qu'il fonctionne sur des données jamais vues.
- **Paramètres** : ce que le modèle apprend seul pendant l'entraînement.
- **Hyperparamètres** : les réglages que l'on choisit **avant** l'entraînement (nombre d'arbres, profondeur maximale…). On les optimise avec `GridSearchCV` ou `RandomizedSearchCV`.
- **Validation croisée** : on entraîne et évalue le modèle plusieurs fois sur des découpages différents des données, pour obtenir un score plus fiable.

**Algorithmes courants :** régression logistique, k plus proches voisins (k-NN), arbre de décision, Random Forest, SVM, gradient boosting.

**Métriques d'évaluation :**

- **Accuracy** : part de bonnes prédictions.
- **Precision** : quand le modèle prédit « positif », a-t-il raison ?
- **Recall** : parmi les vrais positifs, combien sont détectés ?
- **F1-score** : compromis entre precision et recall.
- **Matrice de confusion** : tableau qui croise les prédictions et la réalité.

## 2. Outils

| Outil | Rôle |
|---|---|
| **pandas** | Charger et manipuler les données |
| **scikit-learn** | Modèles, préparation des données, validation croisée, métriques |
| **XGBoost / LightGBM** | Gradient boosting, très performant sur les données en tableau |
| **Matplotlib / Seaborn** | Visualiser les données et les résultats |
| **PyTorch / Hugging Face** | Deep learning, pour le texte et les images |

## 3. Problèmes qu'on peut rencontrer

| Problème | Explication | Solution |
|---|---|---|
| **Surapprentissage** | Le modèle apprend les données d'entraînement par cœur et se trompe sur de nouvelles données. | Modèle plus simple, régularisation, validation croisée |
| **Sous-apprentissage** | Le modèle est trop simple et se trompe partout. | Modèle plus complexe, meilleures variables |
| **Valeurs manquantes** | Cases vides dans les données. | Les remplacer (imputation), par exemple par la médiane |
| **Variables texte** | Les modèles ne comprennent que les nombres. | Les encoder (`OneHotEncoder`) |
| **Échelles différentes** | Des variables comme l'âge et le prix ont des ordres de grandeur différents. | Les standardiser (`StandardScaler`) |
| **Classes déséquilibrées** | Une classe est bien plus fréquente ; l'accuracy devient trompeuse. | `class_weight="balanced"`, F1-score plutôt qu'accuracy |
| **Fuite de données** | Le modèle a vu des informations du jeu de test : ses scores sont faussement bons. | Séparer train/test en premier, préparer les données dans un `Pipeline` |

## 4. Exemple de code commenté

Le jeu de données **Iris** est fourni avec scikit-learn : 150 fleurs, décrites par la longueur et la largeur de leurs sépales et pétales, réparties en 3 espèces. Le but est de prédire l'espèce.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Charger les données : X = les mesures des fleurs, y = l'espèce (0, 1 ou 2)
X, y = load_iris(return_X_y=True)

# Séparer entraînement (80 %) et test (20 %)
# stratify=y garde la même proportion de chaque espèce dans les deux parties
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Le pipeline enchaîne la standardisation des mesures et le modèle k-NN
modele = make_pipeline(StandardScaler(), KNeighborsClassifier())

# Tester plusieurs valeurs de l'hyperparamètre n_neighbors (nombre de voisins)
# en validation croisée sur 5 découpages
grille = {"kneighborsclassifier__n_neighbors": [3, 5, 7, 9]}
recherche = GridSearchCV(modele, grille, cv=5)
recherche.fit(X_train, y_train)
print("Meilleur nombre de voisins :", recherche.best_params_)

# Évaluer le meilleur modèle sur le jeu de test, jamais vu pendant l'entraînement
y_pred = recherche.predict(X_test)
print(classification_report(y_test, y_pred, target_names=load_iris().target_names))
```
