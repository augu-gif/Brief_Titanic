# Brief_Titanic

Contexte du projet

En tant que Data Scientist junior au sein d’un cabinet de conseil en data, vous êtes missionné par un musée maritime. Celui-ci prépare une exposition interactive sur le naufrage du Titanic et souhaite répondre à une question : « Quels facteurs ont réellement influencé les chances de survie des passagers ? »

Le musée vous transmet un fichier CSV issu de ses archives (dataset Titanic, Kaggle). Ce fichier n’est pas exploitable directement :

environ 20 % des âges sont manquants ;

la colonne Cabin est vide à plus de 75 % ;

certaines informations utiles sont cachées dans du texte (le titre Mr, Mrs, Miss ou Master dans le nom des passagers) ;

les variables catégorielles (Sex, Embarked) doivent être encodées ;

certaines colonnes n’apportent aucune information prédictive (PassengerId, Ticket).

Votre mission se déroule en 4 étapes :

Analyse exploratoire (EDA) : comprendre la structure des données, les distributions, les corrélations et les valeurs aberrantes, puis formuler des hypothèses (par exemple : « les femmes et les passagers de 1re classe ont davantage survécu »).

Nettoyage et transformation : traiter les valeurs manquantes en justifiant chaque choix, encoder les variables, créer de nouvelles variables pertinentes (feature engineering : titre, taille de la famille, voyage seul…), 

Modélisation : entraîner et comparer au moins 3 modèles de classification (par exemple régression logistique, Random Forest, XGBoost),

Compétences visées :

C34 : assurer le nettoyage, la transformation et la réduction de la dimensionnalité des datasets

C39 : effectuer l’analyse exploratoire des données

Ressources :

Dataset : https://www.kaggle.com/c/titanic

Documentation : pandas, scikit-learn

Jour 1 (matin) : lancement du brief, veille, découverte du dataset, analyse exploratoire.

Vous avez 1h pour faire une veille présenter ce que c'est que la classification et les bonnes pratiques de nettoyage de données,, ensuite 2 groupes seront choisis au hasard pour faire la présentation de leur veilles.

Jour 1 (après-midi) : nettoyage, traitement des valeurs manquantes, encodage.

Jour 2 (matin) : feature engineering, modélisation et comparaison des modèles.

Jour 2 (après-midi) :  finalisation du dépôt, puis présentation orale à 16h.

Outils :

Python, Jupyter Notebook

pandas, NumPy, scikit-learn, 

Git et GitHub pour le versionnage, avec des commits réguliers et explicites

Accompagnement

Le formateur est disponible en support. Les apprenants doivent d’abord chercher dans la documentation, puis solliciter leurs pairs avant le formateur.

Les notebooks publics Kaggle peuvent servir d’inspiration, mais tout copier-coller non compris et non justifié est proscrit.

Modalités d'évaluation

L’évaluation repose sur trois éléments.
Le dépôt GitHub et le notebook : qualité et justification du nettoyage, pertinence de l’EDA, rigueur de la modélisation, lisibilité du code.
La restitution orale (3 ou 4 groupes choisis au hasard) : 10 minutes de présentation avec Introduction, sommaire, conclusion et numéro de slide. L’apprenant présente ses choix de nettoyage, ses principales découvertes, la comparaison des modèles en s’adressant au client (le musée) dans un langage accessible.

Critères de performance

- Toutes les valeurs manquantes sont traitées et chaque choix (imputation, suppression) est justifié par écrit dans le notebook.
- Les variables catégorielles sont encodées et le dataset final ne contient que des variables numériques exploitables.
- Au moins 2 nouvelles variables sont créées par feature engineering, et leur apport est démontré.
- Au moins 3 modèles de classification sont comparés.
- Le notebook s’exécute de bout en bout sans erreur (reproductibilité, random_state fixé).
- Le README permet à une personne extérieure de comprendre et de relancer le projet.