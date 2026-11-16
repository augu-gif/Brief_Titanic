# 🚢 Brief Titanic

> Quels facteurs ont réellement influencé les chances de survie des passagers du Titanic ?

## Sommaire

- [Contexte du projet](#contexte-du-projet)
- [Les données](#les-données)
- [Mission](#mission)
- [Planning](#planning)
- [Compétences visées](#compétences-visées)
- [Outils](#outils)
- [Accompagnement](#accompagnement)
- [Modalités d'évaluation](#modalités-dévaluation)
- [Critères de performance](#critères-de-performance)
- [Ressources](#ressources)

---

## Contexte du projet

En tant que **Data Scientist junior** au sein d'un cabinet de conseil en data, vous êtes missionné par un **musée maritime**. Celui-ci prépare une exposition interactive sur le naufrage du Titanic et souhaite répondre à une question :

> « Quels facteurs ont réellement influencé les chances de survie des passagers ? »

## Les données

Le musée transmet un fichier CSV issu de ses archives ([dataset Titanic, Kaggle](https://www.kaggle.com/c/titanic)). Ce fichier n'est pas exploitable directement :

| Problème | Détail |
|---|---|
| Valeurs manquantes | Environ **20 %** des âges sont manquants |
| Colonne presque vide | `Cabin` est vide à plus de **75 %** |
| Information cachée dans du texte | Le titre (`Mr`, `Mrs`, `Miss`, `Master`) est dans le nom des passagers |
| Variables catégorielles | `Sex` et `Embarked` doivent être encodées |
| Colonnes sans valeur prédictive | `PassengerId`, `Ticket` |

## Mission

La mission se déroule en 4 étapes :

1. **Analyse exploratoire (EDA)** : comprendre la structure des données, les distributions, les corrélations et les valeurs aberrantes, puis formuler des hypothèses (par exemple : « les femmes et les passagers de 1re classe ont davantage survécu »).
2. **Nettoyage et transformation** : traiter les valeurs manquantes en justifiant chaque choix, encoder les variables, créer de nouvelles variables pertinentes (*feature engineering* : titre, taille de la famille, voyage seul…).
3. **Modélisation** : entraîner et comparer au moins 3 modèles de classification (par exemple régression logistique, Random Forest, XGBoost).
4. **Restitution** : présenter les résultats au client.

## Planning

| Jour | Créneau | Activités |
|---|---|---|
| **Jour 1** | Matin | Lancement du brief, veille, découverte du dataset, analyse exploratoire |
| | Après-midi | Nettoyage, traitement des valeurs manquantes, encodage |
| **Jour 2** | Matin | Feature engineering, modélisation et comparaison des modèles |
| | Après-midi | Finalisation du dépôt, puis **présentation orale à 16 h** |

> **Veille (Jour 1, matin)** : 1 h pour présenter ce qu'est la classification et les bonnes pratiques de nettoyage de données. 2 groupes sont ensuite tirés au sort pour présenter leur veille.

## Compétences visées

- **C34** : assurer le nettoyage, la transformation et la réduction de la dimensionnalité des datasets.
- **C39** : effectuer l'analyse exploratoire des données.

## Outils

- Python, Jupyter Notebook
- pandas, NumPy, scikit-learn
- Git et GitHub pour le versionnage, avec des commits réguliers et explicites

## Accompagnement

Le formateur est disponible en support. Les apprenants doivent d'abord **chercher dans la documentation**, puis **solliciter leurs pairs**, avant de faire appel au formateur.

Les notebooks publics Kaggle peuvent servir d'inspiration, mais tout copier-coller non compris et non justifié est proscrit.

## Modalités d'évaluation

L'évaluation repose sur deux éléments :

1. **Le dépôt GitHub et le notebook** : qualité et justification du nettoyage, pertinence de l'EDA, rigueur de la modélisation, lisibilité du code.
2. **La restitution orale** (3 ou 4 groupes tirés au sort) : 10 minutes de présentation avec introduction, sommaire, conclusion et numéros de slides. L'apprenant présente ses choix de nettoyage, ses principales découvertes et la comparaison des modèles, en s'adressant au client (le musée) dans un langage accessible.

## Critères de performance

- [ ] Toutes les valeurs manquantes sont traitées et chaque choix (imputation, suppression) est justifié par écrit dans le notebook.
- [ ] Les variables catégorielles sont encodées et le dataset final ne contient que des variables numériques exploitables.
- [ ] Au moins 2 nouvelles variables sont créées par feature engineering, et leur apport est démontré.
- [ ] Au moins 3 modèles de classification sont comparés.
- [ ] Le notebook s'exécute de bout en bout sans erreur (reproductibilité, `random_state` fixé).
- [ ] Le README permet à une personne extérieure de comprendre et de relancer le projet.

## Ressources

- Dataset : [Titanic – Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)
- Documentation : [pandas](https://pandas.pydata.org/docs/), [scikit-learn](https://scikit-learn.org/stable/user_guide.html)
