# Guide du Pipeline Machine Learning : Feature Engineering, Modélisation et Comparaison

Le **feature engineering**, la **modélisation** et la **comparaison des modèles** forment les trois piliers successifs d'un pipeline de Machine Learning performant. Une bonne préparation des données simplifie souvent le choix de l'algorithme et maximise la précision des prédictions sur de nouvelles données.

---

## 1. Le Feature Engineering (Ingénierie des caractéristiques)

Cette étape consiste à transformer les données brutes en variables pertinentes qui facilitent l'apprentissage de l'algorithme.

* **Nettoyage et imputation.** Vous devez traiter les valeurs manquantes en les remplaçant par la moyenne, la médiane ou le mode. Il faut aussi détecter et gérer les valeurs aberrantes (outliers) qui faussent les calculs.
* **Encodage des variables catégorielles.** Les algorithmes exigent des valeurs numériques. Utilisez le **One-Hot Encoding** pour les catégories sans ordre hiérarchique et le **Label Encoding** ou l'encodage ordinal lorsqu'un ordre existe.
* **Mise à l'échelle (Scaling).** Les modèles basés sur les distances (KNN, SVM, réseaux de neurones) nécessitent des variables sur une même échelle. Appliquez la **normalisation Min-Max** (valeurs entre 0 et 1) ou la **standardisation Z-score** (moyenne à 0, écart-type à 1).
* **Création et extraction.** Utilisez votre expertise métier pour combiner des variables existantes, par exemple extraire l'heure d'une date complète. Vous pouvez aussi réduire la dimensionnalité avec l'**Analyse en Composantes Principales (ACP)**.
* **Sélection des variables.** Éliminez les données redondantes ou bruitées pour éviter le surapprentissage (overfitting). L'importance par permutation ou les coefficients de corrélation aident à garder uniquement les variables utiles.

---

## 2. La Modélisation

Une fois les données préparées, vous entraînez des algorithmes pour qu'ils apprennent les motifs sous-jacents.

* **Séparation des données.** Divisez toujours votre jeu de données en un ensemble d'entraînement (train) et un ensemble de test (test) avant toute transformation. Cela évite les fuites de données (data leakage) et garantit une évaluation honnête.
* **Choix d'un modèle de référence (Baseline).** Commencez par un modèle simple et rapide comme une régression linéaire ou logistique. Ce score de base sert de point de repère pour mesurer l'apport réel des modèles plus complexes.
* **Validation croisée (Cross-Validation).** Entraînez et évaluez vos modèles sur plusieurs découpages différents des données (comme la méthode K-Fold). Cette pratique réduit la variance de l'évaluation et assure la robustesse du modèle.
* **Optimisation des hyperparamètres.** Ajustez les réglages internes des algorithmes performants (taux d'apprentissage, profondeur des arbres) via une recherche sur grille (Grid Search) ou aléatoire (Random Search).

---

## 3. La Comparaison des Modèles

Comparer les modèles permet de sélectionner celui qui généralise le mieux tout en respectant vos contraintes techniques.

| Famille de modèles | Exemples courants | Points forts | Limites principales |
| :--- | :--- | :--- | :--- |
| **Modèles linéaires** | Régression Linéaire, Logistique | Très rapides, faciles à interpréter | Captent mal les relations non linéaires |
| **Arbres et Ensembles** | Random Forest, XGBoost, LightGBM | Très performants sur données tabulaires, peu sensibles aux échelles | Moins interprétables, risque de surapprentissage |
| **Modèles à noyaux / distance** | SVM, KNN | Efficaces sur de petits jeux de données complexes | Lents sur gros volumes, très sensibles au scaling et outliers |
| **Réseaux de neurones** | MLP, Deep Learning | Excellents sur images, textes et très grands volumes | Exigent beaucoup de données et de puissance de calcul |

### Critères d'évaluation et de choix

* **Métriques de classification.** Si vous prédisez une catégorie, comparez les modèles avec l'**Accuracy** (si les classes sont équilibrées), le **score F1** (compromis entre précision et rappel) ou l'**aire sous la courbe ROC (ROC-AUC)**.
* **Métriques de régression.** Si vous prédisez une valeur numérique, utilisez l'**erreur quadratique moyenne (RMSE)** pour pénaliser les grosses erreurs, ou l'**erreur absolue moyenne (MAE)** pour une lecture directe dans l'unité cible.
* **Critères opérationnels.** Au-delà de la performance pure, comparez le temps d'entraînement, la latence de prédiction et la facilité de déploiement en production.


# Sources/Références :

* **IBM** : https://www.ibm.com/fr-fr/think/topics/feature-engineering
* **Wikipédia** : https://fr.wikipedia.org/wiki/Ing%C3%A9nierie_des_caract%C3%A9ristiques
* **Google for developpers** : https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring?hl=fr
* **Médium**
* **OpenClassRoom**
* **Youtube**

